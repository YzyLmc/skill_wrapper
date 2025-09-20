from gymnasium import Env, spaces
import numpy as np
import os
import cv2

class DorflEnv(Env):
    def __init__(self):
        super().__init__()
        # Initialize your environment here
        self.transition_matrix = None  # Example attribute

        self.observation_space = spaces.Box(low=0, high=255, shape=(2048, 1088), dtype=np.uint8)
        
        # actions:
        # 0: grasp_pb_jar
        # 1: open_pb_jar
        # 2: grasp_knife1
        # 3: scoop
        # 4: spread
        # 5: drop_knife1
        self.action_space = spaces.Discrete(6)

        # states:
        #  The coding is formulated according to this:
        # First digit: bread has pb?
        #   0: has pb
        #   1: no pb
        # Second digit: knife as pb?
        #   0: has pb
        #   1: no pb
        # Third digit: holding knife?
        #   0: knife in cup
        #   1: holding knife
        #   2: knife dropped
        # Forth digit: pb jar open?
        #   0: closed
        #   1: open
        # Fifth digit: pb jar picked up?
        #   0: on table
        #   1: picked up
        self.state = np.array([0, 0, 0, 0, 0])
        
        self._load_images()
    
    def _get_next_state(self, current_state, action):
        # Implement the logic to get the next state based on current state and action
        next_state = current_state.copy()
        success = False
        # Example transition logic (to be replaced with actual logic)
        if action == 0:  # grasp_pb_jar
            if current_state[4] == 0:  # if pb jar is on table
                next_state[4] = 1  # pick up pb jar
                success = True
        elif action == 1:  # open_pb_jar
            if current_state[4] == 1 and current_state[3] == 0:  # if pb jar is picked up and closed
                next_state[3] = 1  # open pb jar
                success = True
        elif action == 2:  # grasp_knife1
            if current_state[2] == 0:  # if knife is in cup
                next_state[2] = 1  # hold knife
                success = True
        elif action == 3:  # scoop
            if current_state[3] == 1 and current_state[2] == 1 and current_state[1] == 0:  # if jar is open, and holding knife, and knife has no pb
                next_state[1] = 1  # knife now has pb
                success = True
        elif action == 4:  # spread
            if current_state[0] == 0 and current_state[2] == 1 and current_state[1] == 1:  # if bread does not have pb and holding knife and knife has pb
                next_state[0] = 1  # bread now has pb
                success = True
        elif action == 5:  # drop_knife1
            if current_state[2] == 1:  # if holding knife
                next_state[2] = 2  # drop knife
                success = True

        return next_state, success


    def _load_images(self):
        dir_name = os.path.join(os.path.dirname(__file__), 'imgs')
        img_files = os.listdir(dir_name)
        self.images = {}
        for img_path in img_files:
            if not img_path.endswith('.jpg'):
                continue
            img_full_path = os.path.join(dir_name, img_path)
            img = cv2.imread(img_full_path)
            self.images[img_path.strip(".jpg")] = img
            print(f"Loaded image: {img_path}")
    
    def _get_curr_image(self):
        # Implement the logic to get the current image based on the state
        key = "".join(map(str, self.state.tolist()))
        return self.images[key]

    def step(self, action):
        # Get next state and success status
        next_state, success = self._get_next_state(self.state, action)
        
        # Update current state
        self.state = next_state
        
        # Get observation (current image)
        observation = self._get_curr_image()
        
        # Define reward based on success and goal completion
        reward = 1.0 if success else 0.0
        
        # Check if episode is terminated (goal reached: bread has pb)
        terminated = (self.state[0] == 1) or (self.state[2] == 2)  # bread has pb or knife dropped
        
        # Truncated is False for this environment (no time limits)
        truncated = False
        
        # Info dictionary
        info = {"success": success, "state": self.state.copy()}
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Reset the environment to initial state
        super().reset(seed=seed)
        self.state = np.array([0, 0, 0, 0, 0])
        observation = self._get_curr_image()
        info = {"success": None, "state": self.state.copy()}
        return observation, info

    def render(self, mode='human'):
        # Get current state image
        img = self._get_curr_image()
        
        if mode == 'human':
            # Display image using OpenCV
            cv2.imshow('DorflEnv', img)
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
    env = DorflEnv()
    
    # Reset environment
    obs, info = env.reset()
    print("Environment initialized. Current state:", env.state)
    print("\nControls:")
    print("0: grasp_pb_jar")
    print("1: open_pb_jar")
    print("2: grasp_knife1")
    print("3: scoop")
    print("4: spread")
    print("5: drop_knife1")
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
            elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
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
                
                if terminated and env.state[0] == 1:
                    print("\nGoal achieved! Bread has peanut butter.")
                elif terminated and env.state[2] == 2:
                    print("\nEpisode terminated. Knife was dropped.")
                
                if terminated:
                    print("Press 'r' to reset or 'q' to quit.")
            else:
                print("Invalid key. Use 0-5 for actions, 'r' to reset, 'q' to quit.")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()

