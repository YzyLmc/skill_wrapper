from gymnasium import Env, spaces
import numpy as np
import os
import cv2

class FrankaEnv(Env):
    def __init__(self, scene_id=1):
        super().__init__()
        
        self.observation_space = spaces.Box(low=0, high=255, shape=(2048, 1088), dtype=np.uint8)
        
        # Actions based on franka.yaml skills:
        # 0: Pick teapot
        # 1: Pick bowl  
        # 2: Pick sponge
        # 3: Place teapot
        # 4: Place bowl
        # 5: Place sponge
        # 6: Stack teapot on plate
        # 7: Stack bowl on plate
        # 8: Stack sponge on plate
        # 9: Pour teapot into mug
        # 10: Wipe plate with sponge
        self.action_space = spaces.Discrete(11)

        # State encoding based on franka README.md (7 bits):
        # bit 1 - gripper: 0=empty, 1=holding teapot, 2=holding bowl, 3=holding sponge
        # bit 2 - teapot: 0=not in scene, 1=origin, 2=on plate, 3=in gripper
        # bit 3 - bowl: 0=not in scene, 1=origin, 2=on plate, 3=in gripper
        # bit 4 - sponge: 0=not in scene, 1=origin, 2=on plate, 3=in gripper
        # bit 5 - plate: 0=not in scene, 1=empty, 2=has teapot, 3=has bowl, 4=has sponge
        # bit 6 - mug_location: 0=not in scene, 1=at origin
        # bit 7 - mug_content: 0=nothing, 1=beans
        self.scene_id = scene_id
        self.state = np.array([0, 1, 1, 0, 1, 0, 0])  # Initial state: all objects at origin
        
        self._load_images()
    
    def _get_next_state(self, current_state, action):
        next_state = current_state.copy()
        success = False
        
        gripper, teapot, bowl, sponge, plate, mug_loc, mug_content = current_state
        
        if action == 0:  # Pick teapot
            if gripper == 0 and teapot == 1:  # gripper empty and teapot at origin
                next_state[0] = 1  # gripper holding teapot
                next_state[1] = 3  # teapot in gripper
                success = True
            elif gripper == 0 and teapot == 2:  # gripper empty and teapot on plate
                next_state[0] = 1  # gripper holding teapot
                next_state[1] = 3  # teapot in gripper
                next_state[4] = 1  # plate becomes empty
                success = True
                
        elif action == 1:  # Pick bowl
            if gripper == 0 and bowl == 1:  # gripper empty and bowl at origin
                next_state[0] = 2  # gripper holding bowl
                next_state[2] = 3  # bowl in gripper
                success = True
            elif gripper == 0 and bowl == 2:  # gripper empty and bowl on plate
                next_state[0] = 2  # gripper holding bowl
                next_state[2] = 3  # bowl in gripper
                next_state[4] = 1  # plate becomes empty
                success = True
                
        elif action == 2:  # Pick sponge
            if gripper == 0 and sponge == 1:  # gripper empty and sponge at origin
                next_state[0] = 3  # gripper holding sponge
                next_state[3] = 3  # sponge in gripper
                success = True
            elif gripper == 0 and sponge == 2:  # gripper empty and sponge on plate
                next_state[0] = 3  # gripper holding sponge
                next_state[3] = 3  # sponge in gripper
                next_state[4] = 1  # plate becomes empty
                success = True
                
        elif action == 3:  # Place teapot
            if gripper == 1 and teapot == 3:  # holding teapot
                next_state[0] = 0  # gripper becomes empty
                next_state[1] = 1  # teapot at origin
                success = True
                
        elif action == 4:  # Place bowl
            if gripper == 2 and bowl == 3:  # holding bowl
                next_state[0] = 0  # gripper becomes empty
                next_state[2] = 1  # bowl at origin
                success = True
                
        elif action == 5:  # Place sponge
            if gripper == 3 and sponge == 3:  # holding sponge
                next_state[0] = 0  # gripper becomes empty
                next_state[3] = 1  # sponge at origin
                success = True
                
        elif action == 6:  # Stack teapot on plate
            if gripper == 1 and teapot == 3 and plate == 1:  # holding teapot and plate empty
                next_state[0] = 0  # gripper becomes empty
                next_state[1] = 2  # teapot on plate
                next_state[4] = 2  # plate has teapot
                success = True
                
        elif action == 7:  # Stack bowl on plate
            if gripper == 2 and bowl == 3 and plate == 1:  # holding bowl and plate empty
                next_state[0] = 0  # gripper becomes empty
                next_state[2] = 2  # bowl on plate
                next_state[4] = 3  # plate has bowl
                success = True
                
        elif action == 8:  # Stack sponge on plate
            if gripper == 3 and sponge == 3 and plate == 1:  # holding sponge and plate empty
                next_state[0] = 0  # gripper becomes empty
                next_state[3] = 2  # sponge on plate
                next_state[4] = 4  # plate has sponge
                success = True
                
        elif action == 9:  # Pour teapot into mug
            if gripper == 1 and teapot == 3 and mug_loc == 1 and mug_content == 0:  # holding teapot, mug at origin and empty
                next_state[6] = 1  # mug gets beans
                success = True
                
        elif action == 10:  # Wipe plate with sponge
            if gripper == 3 and sponge == 3 and plate != 0:  # holding sponge and plate in scene
                # plate becomes empty after wiping
                next_state[4] = 1  # plate empty
                success = True

        return next_state, success

    def _load_images(self):
        dir_name = os.path.join(os.path.dirname(__file__), 'imgs')
        img_files = []
        
        # Check all subdirectories for images
        for root, dirs, files in os.walk(dir_name):
            for file in files:
                if file.endswith('.jpg'):
                    img_files.append(os.path.join(root, file))
        
        self.images = {}
        for img_path in img_files:
            img = cv2.imread(img_path)
            if img is not None:
                # Extract filename without extension as key
                filename = os.path.basename(img_path).replace('.jpg', '')
                self.images[filename] = img
                print(f"Loaded image: {filename}")
    
    def _get_curr_image(self):
        # Convert state to string key for image lookup
        key = "".join(map(str, self.state.tolist()))
        
        # If exact state image doesn't exist, try to find closest match
        if key in self.images:
            return self.images[key]
        else:
            # Fallback: return first available image or create dummy image
            raise ValueError(f"No image found for state key: {key}")

    def step(self, action):
        # Get next state and success status
        next_state, success = self._get_next_state(self.state, action)
        
        # Update current state
        self.state = next_state
        
        # Get observation (current image)
        observation = self._get_curr_image()
        
        # Define reward based on success
        reward = 1.0 if success else 0.0
        
        # Episode termination logic (customize as needed)
        terminated = False  # Disable termination for data collection
        
        # Truncated is False for this environment (no time limits)
        truncated = False
        
        # Info dictionary
        info = {"success": success, "state": self.state.copy()}
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Reset the environment to initial state
        super().reset(seed=seed)
        if self.scene_id == 1:
            self.state = np.array([0, 1, 1, 0, 1, 0, 0])
        else:
            raise ValueError("Unsupported scene_id. Currently only scene_id=1 is supported.")
        observation = self._get_curr_image()
        info = {"success": None, "state": self.state.copy()}
        return observation, info

    def render(self, mode='human'):
        # Get current state image
        img = self._get_curr_image()
        
        if mode == 'human':
            # Display image using OpenCV
            cv2.imshow('FrankaEnv', img)
            cv2.waitKey(1)  # Non-blocking wait
        elif mode == 'rgb_array':
            # Return the image as RGB array
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img

    def close(self):
        # Close OpenCV windows
        cv2.destroyAllWindows()


def main():
    # Create environment
    env = FrankaEnv()
    
    # Reset environment
    obs, info = env.reset()
    print("Environment initialized. Current state:", env.state)
    print("State encoding: [gripper, teapot, bowl, sponge, plate, mug_location, mug_content]")
    print("\nControls:")
    print("0: Pick teapot")
    print("1: Pick bowl")
    print("2: Pick sponge")
    print("3: Place teapot")
    print("4: Place bowl")
    print("5: Place sponge")
    print("6: Stack teapot on plate")
    print("7: Stack bowl on plate")
    print("8: Stack sponge on plate")
    print("9: Pour teapot into mug")
    print("a: Wipe plate with sponge")
    print("q: quit")
    print("r: reset")
    print("\nPress keys to control the robot...")
    
    # Render initial state
    env.render()
    
    try:
        while True:
            # Get keyboard input
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == 27:  # q or ESC to quit
                break
            elif key == ord('r'):  # r to reset
                obs, info = env.reset()
                print("\nEnvironment reset. Current state:", env.state)
                env.render()
                continue
            elif key in [ord(str(i)) for i in range(10)]:
                action = int(chr(key))
                print(f"\nAction: {action}")
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Print results
                print(f"New state: {env.state}")
                print(f"Success: {info['success']}")
                print(f"Reward: {reward}")
                print(f"Terminated: {terminated}")
                
                # Render new state
                env.render()
                
            elif key == ord('a'):  # Wipe action (action 10)
                action = 10
                print(f"\nAction: {action} (Wipe)")
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Print results
                print(f"New state: {env.state}")
                print(f"Success: {info['success']}")
                print(f"Reward: {reward}")
                print(f"Terminated: {terminated}")
                
                # Render new state
                env.render()
            else:
                print("Invalid key. Use 0-9 for actions, 'a' for wipe, 'r' to reset, 'q' to quit.")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()