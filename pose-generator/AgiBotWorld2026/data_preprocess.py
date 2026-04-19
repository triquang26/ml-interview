import os
import json
import pandas as pd
import numpy as np
import cv2
import pinocchio as pin
import imageio

START_EPISODE = 0
END_EPISODE = 105
BASE_DIR = "/mnt/data/sftp/data/quangpt3/difgit/ml-interview/pose-generator/AgiBotWorld2026"
URDF_PATH = f"{BASE_DIR}/genie_sim/source/teleop/app/share/genie_robot_description/urdf/G2/G2.urdf"
INFO_PATH = f"{BASE_DIR}/data/meta/info.json"

STICK_LENGTH = 0.10 # 10cm extension

def process_episode(episode):
    PARQUET_PATH = f"{BASE_DIR}/data/data/chunk-000/episode_{episode:06d}.parquet"
    VIDEO_PATH = f"{BASE_DIR}/data/videos/chunk-000/observation.images.top_head/episode_{episode:06d}.mp4"
    OUTPUT_DIR = f"{BASE_DIR}/extracted_data/extracted_poses_ep_{episode:06d}"
    
    if not os.path.exists(PARQUET_PATH) or not os.path.exists(VIDEO_PATH):
        print(f"Skipping Episode {episode}: Missing video or parquet file.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    POSE_DIR = os.path.join(OUTPUT_DIR, "pose")
    FRAME_DIR = os.path.join(OUTPUT_DIR, "frame")
    os.makedirs(POSE_DIR, exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)
    
    # 1. Load Camera
    with open(INFO_PATH, 'r') as f:
        info = json.load(f)
    cam_params = info["camera_parameters"]["0"]["intrinsic_head_front_rgb"]
    K = np.array([[cam_params["Fx"], 0, cam_params["Cx"]],
                  [0, cam_params["Fy"], cam_params["Cy"]],
                  [0, 0, 1]], dtype=np.float32)

    # 2. Load State Data
    df = pd.read_parquet(PARQUET_PATH)

    # 3. Build Kinematics
    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()
    
    # 4. Load Video to get frames length and dimension
    reader = imageio.get_reader(VIDEO_PATH)
    video_n_frames = reader.count_frames()
    
    # Safely get the max number of frames that are available in both video and parquet
    n_frames = min(video_n_frames, len(df))
    if n_frames == 0:
        print(f"Skipping Episode {episode}: No valid frames.")
        return
        
    # get first frame for dimension
    sample_frame = reader.get_data(0)
    h, w = sample_frame.shape[:2]

    print(f"--- Processing Episode {episode} ---")
    print(f"Total synced frames: {n_frames} (Video: {video_n_frames}, Parquet: {len(df)})")
    
    # Uniformly sample 50 frames
    sampled_indices = np.linspace(0, n_frames - 1, min(50, n_frames), dtype=int)
    print(f"Extracting {len(sampled_indices)} frames...")

    for frame_idx in sampled_indices:
        state = np.array(df['observation.state'].iloc[frame_idx])
        T_base_cam = pin.SE3(state[126:135].reshape(3, 3), state[156:159])

        q = pin.neutral(model)
        scale = (np.pi / 180.0) if np.max(np.abs(state[40:54])) > 6.28 else 1.0

        # Map all joints
        mapping = {}
        for i in range(1, 8): 
            mapping[f"idx2{i}_arm_l_joint{i}"] = state[39 + i]
            mapping[f"idx6{i}_arm_r_joint{i}"] = state[46 + i]
        for i in range(1, 6): 
            mapping[f"idx0{i}_body_joint{i}"] = state[84 + i]

        for jname, val in mapping.items():
            if model.existJointName(jname):
                jid = model.getJointId(jname)
                q[model.joints[jid].idx_q] = val * scale

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        # 5. Create Black Background (same dimensions as frame)
        black_frame = np.zeros((h, w, 3), dtype=np.uint8)

        # 6. Points to Project
        links_body = ["base_link", "body_link1", "body_link2", "body_link3", "body_link4", "body_link5"]
        links_l = ["arm_base_link", "arm_l_link1", "arm_l_link4", "arm_l_link7", "arm_l_end_link"]
        links_r = ["arm_base_link", "arm_r_link1", "arm_r_link4", "arm_r_link7", "arm_r_end_link"]
        
        projected = {}
        
        def get_2d(T_base_pt):
            T_cam_pt = T_base_cam.inverse() * T_base_pt
            pt = T_cam_pt.translation
            if pt[2] > 0.05:
                u = int((K[0,0] * pt[0] / pt[2]) + K[0,2])
                v = int((K[1,1] * pt[1] / pt[2]) + K[1,2])
                return (u, v)
            return None

        # Project standard links
        for name in (links_body + links_l + links_r):
            fid = model.getFrameId(name)
            if fid < len(model.frames):
                pix = get_2d(data.oMf[fid])
                if pix:
                    projected[name] = pix
                    cv2.circle(black_frame, pix, 4, (0, 255, 255), -1)

        for side in ['l', 'r']:
            prev_fid = model.getFrameId(f"arm_{side}_link7")
            last_fid = model.getFrameId(f"arm_{side}_end_link")
            
            if prev_fid < len(model.frames) and last_fid < len(model.frames):
                P_prev = data.oMf[prev_fid].translation
                P_last = data.oMf[last_fid].translation
                
                # Vector of the last segment
                vec = P_last - P_prev
                dist = np.linalg.norm(vec)
                
                if dist > 0:
                    # Extend the same vector by STICK_LENGTH
                    direction = vec / dist
                    P_stick = P_last + (direction * STICK_LENGTH)
                    
                    # Project the new extended point
                    pix_stick = get_2d(pin.SE3(np.eye(3), P_stick))
                    if pix_stick:
                        projected[f"{side}_stick"] = pix_stick
                        cv2.circle(black_frame, pix_stick, 6, (255, 0, 255), -1)

        # Draw Chains
        def draw(chain, color, thick=2):
            for i in range(len(chain)-1):
                if chain[i] in projected and chain[i+1] in projected:
                    cv2.line(black_frame, projected[chain[i]], projected[chain[i+1]], color, thick)

        # Body lines (White)
        draw(links_body, (255, 255, 255), 3)
        draw(["body_link5", "arm_base_link"], (255, 255, 255), 2)
        
        # Arms + The Extended Stick (Follows existing direction)
        draw(links_l + ["l_stick"], (0, 0, 255), 2)
        draw(links_r + ["r_stick"], (0, 255, 0), 2)

        orig_frame = cv2.cvtColor(reader.get_data(frame_idx), cv2.COLOR_RGB2BGR)
        
        # Save Original Video Frame
        frame_out_path = os.path.join(FRAME_DIR, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(frame_out_path, orig_frame)

        # Save Skeleton Pose on Black Background
        pose_out_path = os.path.join(POSE_DIR, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(pose_out_path, black_frame)
        
    print(f"Done processing Episode {episode}!")

def extract_all_poses():
    print(f"Starting extraction for episodes {START_EPISODE} to {END_EPISODE}...")
    for episode in range(START_EPISODE, END_EPISODE + 1):
        try:
            process_episode(episode)
        except Exception as e:
            print(f"Error processing episode {episode}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    extract_all_poses()
