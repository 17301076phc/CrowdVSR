import argparse

parser = argparse.ArgumentParser()

# video source
parser.add_argument('--lr_video', type=str, default='video/diving_cmp_x2.mp4', help='Low resolution video')
# parser.add_argument('--lr_video', type=str, default='video/diving_cmp_x3.mp4', help='Low resolution video')
# parser.add_argument('--lr_video', type=str, default='video/diving_cmp_x4.mp4', help='Low resolution video')
parser.add_argument('--hr_video', type=str, default='video/diving_cmp.mp4', help='High resolution video')
# parser.add_argument('--lr_video', type=str, default='video/musician_cmp_x2.mp4', help='Low resolution video')

# parser.add_argument('--hr_video', type=str, default='video/musician_cmp.mp4', help='High resolution video')

parser.add_argument('--candidates', type=str, default='candidates', help='Candidate video frame')
# parser.add_argument('--candidates', type=str, default='div2k_dataset', help='Candidate video frame')
# parser.add_argument('--sample_interval', type=int, default=23, help='High resolution video')
parser.add_argument('--sample_interval', type=int, default=5, help='High resolution video')

# webrtc
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for HTTP server (default: 0.0.0.0)')
parser.add_argument('--port', type=int, default=8080, help='Port for HTTP server (default: 8080)')

# model
parser.add_argument('--patch_size', type=int, default=48, help='Patch size')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

args = parser.parse_args()
