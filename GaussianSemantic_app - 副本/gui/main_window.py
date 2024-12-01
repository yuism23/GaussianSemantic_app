# gui/main_window.py

import logging
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap
import torch
import threading
from collections import deque
import queue
import torchvision
import os
import time
import math

from .ui_main_window import Ui_MainWindow
from render import Render, GaussianModel, load_camera_model
from render.camera_pose import load_camera_model_from_colmap  
from utils import encode_image_to_base64

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Communicate(QObject):
    render_done = pyqtSignal(str)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.comm = Communicate()
        self.comm.render_done.connect(self.on_render_done)
        
        self.render_queue = deque(maxlen=10)
        self.render_done_event = threading.Event()
        self.latest_pose = None
        self.pose_lock = threading.Lock()
        self.count = 0
        
        self.gaussian_model = None
        self.render = None
        
        self.ui.loadButton.clicked.connect(self.load_point_cloud)
        self.ui.renderButton.clicked.connect(self.render_image)
        self.ui.trainButton.clicked.connect(self.start_training)
        
        self.running = True
        self.render_thread = threading.Thread(target=self.rendering_thread, daemon=True)
        self.render_thread.start()
        
        self.camera_speed = 0.1    
        
        self.setFocusPolicy(Qt.StrongFocus)
        
    def closeEvent(self, event):
        self.running = False
        self.render_done_event.set()  # Trigger the thread to detect the exit signal
        self.render_thread.join()  # Wait for the thread to exit safely
        event.accept()
    
    def load_point_cloud(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Point Cloud PLY File", "", "PLY Files (*.ply);;All Files (*)", options=options)
        if file_name:
            try:
                logging.debug(f"Loading point cloud from: {file_name}")
                self.gaussian_model = GaussianModel(file_name)
                bg_color = torch.tensor([0, 0, 0], dtype=torch.float32).cuda()
                
                camera_intrinsic = "1 PINHOLE 1960 1080 1316 1316 980 540"
                camera_extrinsic = "274 0.99991973151466196 -0.0006936684064747029 -0.0094769649404623044 -0.0083807211685744259 -0.99039223249943242 -0.086686021877042219 0.11989225602179382 1 DJI_20231218125136_0036_Zenmuse-L1-mission.JPG"
                
                default_camera_pose = load_camera_model_from_colmap(camera_intrinsic, camera_extrinsic)
                
                logging.debug(f"Default camera pose:\n{default_camera_pose.world_view_transform}")
                
                self.render = Render(
                    pc=self.gaussian_model,
                    bg_color=bg_color,
                    default_camera_pose=default_camera_pose
                )
                self.ui.imageLabel.setText("Point Cloud Loaded. Ready to Render.")
                logging.debug("Point cloud loaded and render initialized successfully.")
            except Exception as e:
                self.ui.imageLabel.setText(f"Failed to load point cloud: {e}")
                logging.error(f"Error loading point cloud: {e}")

    def render_image(self):
        if self.gaussian_model is None or self.render is None:
            self.ui.imageLabel.setText("Please load a point cloud first.")
            return

        try:
             # Get the camera pose entered by the user
            position = torch.tensor([
                self.ui.posXSpinBox.value(),
                self.ui.posYSpinBox.value(),
                self.ui.posZSpinBox.value()
            ], dtype=torch.float32).cuda()

            orientationQuat = torch.tensor([
                self.ui.orientWSpinBox.value(),
                self.ui.orientXSpinBox.value(),
                self.ui.orientYSpinBox.value(),
                self.ui.orientZSpinBox.value()
            ], dtype=torch.float32).cuda()


            # Create a new camera pose model
            new_camera_pose = load_camera_model(position, orientationQuat)


            # Update the latest camera pose
            with self.pose_lock:
                self.latest_pose = (position, orientationQuat)

            # Trigger rendering
            self.render_done_event.set()
            logging.debug("Render event set.")
        except Exception as e:
            self.ui.imageLabel.setText(f"Error in rendering: {e}")
            logging.error(f"Error in render_image: {e}")

    def training_thread(self):
        for epoch in range(1, 101):
            # Simulate training steps
            time.sleep(0.05)  # Replace with actual training code

            # Update progress bar
            self.ui.progressBar.setValue(epoch)

            if epoch % 10 == 0:
                logging.debug(f"Epoch {epoch}/100 completed.")

        self.ui.imageLabel.setText("Training Completed.")
        logging.debug("Training completed.")

    def start_training(self):
        if self.gaussian_model is None:
            self.ui.imageLabel.setText("Please load a point cloud first.")
            return
        self.train_thread = threading.Thread(target=self.training_thread, daemon=True)
        self.train_thread.start()

    def quaternion_to_rotation_matrix(self, q):
        """
        Convert quaternion to rotation matrix
        """
        w, x, y, z = q
        rot_matrix = torch.tensor([
            [1 - 2 * y**2 - 2 * z**2,     2 * x * y - 2 * w * z,     2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z,       1 - 2 * x**2 - 2 * z**2,   2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y,       2 * y * z + 2 * w * x,     1 - 2 * x**2 - 2 * y**2]
        ], dtype=torch.float32).cuda()
        return rot_matrix

    def rotation_matrix_to_quaternion(self, R):
        """
        Convert rotation matrix to quaternion
        """
        trace = R[0,0] + R[1,1] + R[2,2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2,1] - R[1,2]) * s
            y = (R[0,2] - R[2,0]) * s
            z = (R[1,0] - R[0,1]) * s
        elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
        quat = torch.tensor([w, x, y, z], dtype=torch.float32).cuda()
        return quat



    def update_camera_info(self, position, orientationQuat):
        """
        Update camera information displayed on the UI and log the details.
        
        Args:
            position (torch.Tensor): Camera position in world coordinates, shape [3].
            orientationQuat (torch.Tensor): Camera quaternion representation, shape [4].
        """
        camera_center = position.clone()  # Directly use position as camera_center
        pos_str = f"Position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
        angle_str = self.quaternion_to_euler(orientationQuat)
        orient_str = f"Orientation: (Yaw: {angle_str[0]:.2f}°, Pitch: {angle_str[1]:.2f}°, Roll: {angle_str[2]:.2f}°)"
        logging.debug(f"Camera Info Updated: {pos_str} | {orient_str}")
        logging.debug(f"Camera Center: {camera_center}")

        self.ui.cameraInfoLabel.setText(f"{pos_str} | {orient_str}")

    
    
    def rendering_thread(self):
        logging.debug("Rendering thread started.")
        while self.running:
            self.render_done_event.wait()
            if not self.running:
                break
            self.render_done_event.clear()

            # Get the latest camera pose
            with self.pose_lock:
                if self.latest_pose is None:
                    logging.debug("No latest pose found. Continuing.")
                    continue
                position, orientationQuat = self.latest_pose

            # Create a new camera pose model
            new_camera_pose = load_camera_model(position, orientationQuat)
            logging.debug(f"Rendering with camera pose:\n{new_camera_pose.world_view_transform}")

            # Update the renderer's camera pose
            self.render.update_camera_pose(new_camera_pose)
            logging.debug("Renderer's camera pose updated.")

            # Perform rendering
            try:
                with torch.no_grad():
                    rendered_output = self.render.forward(new_camera_pose)
                    rendered_image = rendered_output

                    # Ensure the images folder exists
                    if not os.path.exists('images'):
                        os.makedirs('images')

                    # Save the rendered image
                    rendered_image_path = f'images/rendered_image_{self.count}.jpg'
                    self.count += 1
                    torchvision.utils.save_image(rendered_image, rendered_image_path)
                    logging.debug(f"Rendered image saved at: {rendered_image_path}")

                # Clear CUDA memory
                del rendered_output, rendered_image
                torch.cuda.empty_cache()

                # Put the rendered image path into the queue
                self.render_queue.append(rendered_image_path)

                if len(self.render_queue) > 9:
                    old_image_path = self.render_queue.popleft()  
                    os.remove(old_image_path)  
                    logging.debug(f"Deleted old image: {old_image_path}")
                
                # Emit the render done signal
                self.comm.render_done.emit(rendered_image_path)
                logging.debug("Render done signal emitted.")

                # Update the camera information display
                self.update_camera_info(position, orientationQuat)
            except Exception as e:
                self.ui.imageLabel.setText(f"Rendering failed: {e}")
                logging.error(f"Error in rendering_thread: {e}")

    def on_render_done(self, image_path):
        try:
            # Load and display the image
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                self.ui.imageLabel.setText("Failed to load rendered image.")
                logging.error(f"Failed to load rendered image from: {image_path}")
            else:
                self.ui.imageLabel.setPixmap(pixmap.scaled(self.ui.imageLabel.size(), Qt.KeepAspectRatio))
                logging.debug(f"Rendered image displayed: {image_path}")
        except Exception as e:
            self.ui.imageLabel.setText(f"Error displaying image: {e}")
            logging.error(f"Error in on_render_done: {e}")

    def quaternion_to_euler(self, qvec):
        """
        Convert quaternion to Euler angles (yaw, pitch, roll).
        Args:
            qvec (torch.Tensor): Quaternion, shape [4], format [w, x, y, z].
        Returns:
            list: Euler angles [yaw, pitch, roll].
        """
        w, x, y, z = qvec
        # Calculate yaw (heading)
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        # Calculate pitch (elevation)
        pitch = math.asin(2.0 * (w * y - z * x))
        # Calculate roll (rotation around forward axis)
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        
        # Convert radians to degrees
        yaw = math.degrees(yaw)
        pitch = math.degrees(pitch)
        roll = math.degrees(roll)
        
        return [yaw, pitch, roll]

    # Add keyboard event handling
    def keyPressEvent(self, event):
        if self.gaussian_model is None or self.render is None:
            return

        key = event.key()
        moved = False

        with self.pose_lock:
            # Use latest_pose if available, else use the current camera pose from the renderer
            if self.latest_pose:
                position, orientationQuat = self.latest_pose
            else:
                position = self.render.camera_model.camera_center.clone()
                # Extract quaternion from world_view_transform (assumes rotation is in the first three columns)
                rot_matrix = self.render.camera_model.world_view_transform[:3, :3]
                orientationQuat = self.rotation_matrix_to_quaternion(rot_matrix).cuda()

            position = position.clone()
            orientation = self.quaternion_to_euler(orientationQuat)

            if key == Qt.Key_W:
                position[1] += self.camera_speed
                moved = True
            elif key == Qt.Key_S:
                position[1] -= self.camera_speed
                moved = True
            elif key == Qt.Key_A:
                position[0] += self.camera_speed
                moved = True
            elif key == Qt.Key_D:
                position[0] -= self.camera_speed
                moved = True
            elif key == Qt.Key_Q:
                position[2] += self.camera_speed
                moved = True
            elif key == Qt.Key_E:
                position[2] -= self.camera_speed
                moved = True
            elif event.key() == Qt.Key_Up:
                orientation[1] += 5  # Increase pitch
                orientationQuat = self.euler_to_quaternion(orientation)
                moved = True
            elif event.key() == Qt.Key_Down:
                orientation[1] -= 5  # Decrease pitch
                orientationQuat = self.euler_to_quaternion(orientation)
                moved = True
            elif event.key() == Qt.Key_Left:
                orientation[0] -= 5  # Turn left (yaw)
                orientationQuat = self.euler_to_quaternion(orientation)
                moved = True
            elif event.key() == Qt.Key_Right:
                orientation[0] += 5  # Turn right (yaw)
                orientationQuat = self.euler_to_quaternion(orientation)
                moved = True
            elif event.key() == Qt.Key_Z:
                orientation[2] -= 5  # Rotate clockwise (roll)
                orientationQuat = self.euler_to_quaternion(orientation)
                moved = True
            elif event.key() == Qt.Key_X:
                orientation[2] += 5  # Rotate counter-clockwise (roll)
                orientationQuat = self.euler_to_quaternion(orientation)
                moved = True

            if moved:
                # Update the latest camera pose
                self.latest_pose = (position, orientationQuat)
                self.render_done_event.set()
                logging.debug(f"Camera moved: Position={position}, OrientationQuat={orientationQuat}")

        super(MainWindow, self).keyPressEvent(event)

    def euler_to_quaternion(self, camera_angle):
        """
        Convert Euler angles [yaw, pitch, roll] to quaternion.
        Args:
            camera_angle (list): Euler angles [yaw, pitch, roll] in degrees.
        Returns:
            torch.Tensor: Quaternion, shape [4], format [w, x, y, z].
        """
        yaw, pitch, roll = camera_angle
        # Convert degrees to radians
        yaw = math.radians(yaw)
        pitch = math.radians(pitch)
        roll = math.radians(roll)
        
        # Compute quaternion
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return torch.tensor([w, x, y, z])
    
