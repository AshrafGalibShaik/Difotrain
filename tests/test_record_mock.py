import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import io

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from capture import record_pose

class TestRecordPose(unittest.TestCase):
    @patch('capture.record_pose.cv2')
    @patch('capture.record_pose.vision')
    @patch('capture.record_pose.python')
    @patch('capture.record_pose.mp')
    def test_main_runs_and_saves_file(self, mock_mp, mock_python, mock_vision, mock_cv2):
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        # Simulate: open -> true, read frame 1 -> true, read frame 2 -> true, read frame 3 -> False (end)
        mock_cap.isOpened.side_effect = [True, True, True, True, False]
        # capture.read() returns (ret, frame)
        mock_cap.read.side_effect = [
            (True, MagicMock()), 
            (True, MagicMock()), 
            (False, None)
        ]
        
        # Mock PoseLandmarker
        mock_landmarker = MagicMock()
        mock_vision.PoseLandmarker.create_from_options.return_value = mock_landmarker
        
        # Mock Detection Result
        mock_result = MagicMock()
        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0
        
        # 33 landmarks for index 0 person
        # The result.pose_landmarks is a list of lists of landmarks
        mock_result.pose_landmarks = [[mock_landmark] * 33]
        
        mock_landmarker.detect_for_video.return_value = mock_result

        # Stub cv2.waitKey to not block
        mock_cv2.waitKey.return_value = 0
        
        # Mock Path to exist for model check
        with patch('capture.record_pose.Path.exists', return_value=True):
            # Capture stdout to avoid clutter
            with patch('sys.stdout', new=io.StringIO()) as fake_out:
                 record_pose.main()

        # Check if file creation code was reached.
        # We check if path was resolved and opened.
        # Ideally we could mock open, but existing check is fine for now if we assume logic flow.
        self.assertTrue(os.path.exists('storage/human_trajectory.json'))
        
        # Clean up
        if os.path.exists('storage/human_trajectory.json'):
            os.remove('storage/human_trajectory.json')

if __name__ == '__main__':
    unittest.main()
