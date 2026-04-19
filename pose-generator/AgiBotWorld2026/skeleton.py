import json
import pandas as pd
import numpy as np
import cv2
import pinocchio as pin
import imageio

EPISODE = 31
FRAME_IDX = 150
BASE_DIR = "/mnt/data/sftp/data/quangpt3/difgit/ml-interview/pose-generator/AgiBotWorld2026"
URDF_PATH = f"{BASE_DIR}/genie_sim/source/teleop/app/share/genie_robot_description/urdf/G2/G2.urdf"
INFO_PATH = f"{BASE_DIR}/data/meta/info.json"
PARQUET_PATH = f"{BASE_DIR}/data/data/chunk-000/episode_{EPISODE:06d}.parquet"
VIDEO_PATH = f"{BASE_DIR}/data/videos/chunk-000/observation.images.top_head/episode_{EPISODE:06d}.mp4"

STICK_LENGTH = 0.10 # 10cm extension

def project_extended_skeleton():
    # 1. Load Camera & State
    with open(INFO_PATH, 'r') as f:
        info = json.load(f)
    cam_params = info["camera_parameters"]["0"]["intrinsic_head_front_rgb"]
    K = np.array([[cam_params["Fx"], 0, cam_params["Cx"]],
                  [0, cam_params["Fy"], cam_params["Cy"]],
                  [0, 0, 1]], dtype=np.float32)

    df = pd.read_parquet(PARQUET_PATH)
    state = np.array(df['observation.state'].iloc[FRAME_IDX])
    T_base_cam = pin.SE3(state[126:135].reshape(3, 3), state[156:159])

    # 2. Build Kinematics
    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()
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

    # 3. Load Image
    reader = imageio.get_reader(VIDEO_PATH)
    n_frames = reader.count_frames()

    print(f"--- Đang xử lý Episode {EPISODE} ---")
    print(f"Video dài: {n_frames} frames")
    print(f"Đang trích xuất Frame index: {FRAME_IDX}")

    if FRAME_IDX >= n_frames:
        print("Cảnh báo: FRAME_IDX vượt quá tổng số frame!")
    frame = cv2.cvtColor(reader.get_data(FRAME_IDX), cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]

    # 4. Points to Project
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
                cv2.circle(frame, pix, 4, (0, 255, 255), -1)

    # --- THE LOGIC: EXTEND CURRENT DIRECTION ---
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
                    cv2.circle(frame, pix_stick, 6, (255, 0, 255), -1)

    # 5. Draw Chains
    def draw(chain, color, thick=2):
        for i in range(len(chain)-1):
            if chain[i] in projected and chain[i+1] in projected:
                cv2.line(frame, projected[chain[i]], projected[chain[i+1]], color, thick)

    # Body lines (White)
    draw(links_body, (255, 255, 255), 3)
    draw(["body_link5", "arm_base_link"], (255, 255, 255), 2)
    
    # Arms + The Extended Stick (Follows existing direction)
    draw(links_l + ["l_stick"], (0, 0, 255), 2)
    draw(links_r + ["r_stick"], (0, 255, 0), 2)

    cv2.imwrite("EXTENDED_DIRECTION.jpg", frame)
    print("Done! Skeleton extended along forearm direction.")

if __name__ == "__main__":
    project_extended_skeleton()