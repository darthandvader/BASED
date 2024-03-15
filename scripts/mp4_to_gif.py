from moviepy.editor import VideoFileClip
videoClip = VideoFileClip("logs/cutting_tissues_twice/nerf_onlytest_time_127000/novel_fast.mp4")
videoClip.write_gif("logs/cutting_tissues_twice/nerf_onlytest_time_127000/novel_fast.gif")

