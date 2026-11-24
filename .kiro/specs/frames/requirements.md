# Requirements Document

## Introduction

This spec builds the Frames_Step, the first step of the Pipeline. It reads the part of the video the person chose, scores every frame, cuts the still parts off the ends, drops the jolted frames, and keeps a few hundred sharp frames spread evenly over the camera's movement. It also finds the pixels that stay the same in every frame, like a drone propeller or a watermark, and makes a Mask that tells the next steps to ignore them. It writes the Picks, their times and the Mask to its Work_Folder, `work/frames/`, where the Camera_Solve_Step reads them.

The Frames_Step is a Required_Step and the first entry in the Step_List. It uses the Job_Folder, the Job_Record's Trim, the Step_Runner, the Step_Report and the Presets from the pipeline-core spec, and it's built like the Fake_Step.

Later the step runs on the Job's cloud machine, which needs at least 16 vCPUs and 16 GB of memory for it. The step itself never checks the machine, so it runs the same on a laptop, and all its maths is tested on the laptop with tiny made-up frames. The decoding comes from the ffmpeg command line tool. numpy and OpenCV do the maths on frames that ffmpeg has already decoded.

Out of scope:

- The Camera_Solve_Step. This spec only writes the files it reads and states that contract in the Pipeline_Doc.
- The cloud wrapper, which reads the Machine_Needs from Config and picks the machine. Nothing in this spec runs anything paid.
- How COLMAP and training use the Mask. This spec writes the Mask in the form both need and says so in the Pipeline_Doc.
- GPU work. Nothing here needs a GPU, torch, gsplat or AWS.
- Changing Skeleton modules beyond the Step_List and its comment in `splatpipe/run.py`, and `splatpipe/probe.py` if the design has the Video_Probe read the colour transfer (A7).

### Decisions made before this draft

The user chose these when asked, before the requirements were written.

- **D1 (Requirement 11):** The Frames_Step goes into the Step_List as its first entry and as a Required_Step: `STEP_LIST = (StepEntry("frames", "splatpipe.frames", True),)`. Until the camera solve spec adds its entry, a Job counts as done once the Frames_Step is done. Every place that assumes an empty Step_List changes in the same work: `splatpipe/run.py`, `tests/test_run.py`, `tests/test_imports.py`, `tests/test_docs.py`, `tests/conftest.py`, the Pipeline_Doc's "The step list" section, and a note under the pipeline-core spec's Requirement 8, criterion 2, which said the list starts empty. A Step_List the caller gives empty is still refused.
- **D2 (Requirement 6):** A Shared_Setting `min_picks` sets the fewest Picks a Job can go on with. With fewer Picks, the step fails with an Expected_Failure that says what to film differently. A clip that is still from start to end is kept whole by the Still_Trim, so the Still_Trim never removes everything, and such a clip fails only if it gives too few Picks.
- **D3 (Requirement 7):** The Mask comes from up to 240 Score_Frames of the Kept_Range, chosen as A15 says. The review changed how they're chosen: the draft spread exactly 240 evenly over the Kept_Range. For each pixel, the step averages its change from one sample to the next. A pixel is stuck when that average is under 0.2 times the median of every pixel's average. The step uses the Mask only when the stuck pixels cover 0.2% to 25% of the frame, and does no clean-up on it. The Mask is one black-and-white PNG for every Pick, at the Pick_Size, where black means ignore. COLMAP takes it as its single camera mask (`--ImageReader.camera_mask_path`), and training reads the same file.
- **D4 (Requirements 1 and 9):** `min_vcpus = 16` and `min_memory_gb = 16` go in `[frames]` for the cloud spec to read, and the Pipeline_Doc states them. The step never checks the machine. The design keeps peak memory under 16 GB for a 10-minute 4K clip by streaming frames from ffmpeg and never holding every frame.

### Decisions made at review

The user reviewed the draft and chose these.

- **D5 (Requirements 1 and 12):** The step needs ffmpeg and ffprobe 5.1 or newer, because it uses ffmpeg's `-fps_mode` option. When ffmpeg's output can't be read, the Readable_Error says to install ffmpeg 5.1 or newer. The Pipeline_Doc's frames section says the Job's machine image needs ffmpeg 5.1 or newer. Ubuntu 24.04's packaged ffmpeg is 6.1, which is new enough.

The review also confirmed every assumption below, with A12, A13 and A15 changed as they now read, and left O1 open for Samarth.

### Assumptions confirmed at review

The user didn't decide these before the draft. Each one is marked in the requirement it belongs to. The user confirmed all of them at review.

- **A1 (Requirement 4):** A Stretch of frames that aren't Still_Frames and lasts less than `blip_seconds` (0.5 s, confirmed by the user at review) is a Blip and counts as still, so a Blip inside a still Stretch doesn't end the still Stretch.
- **A2 (Requirements 4 and 5):** The 1 s window for the Still_Trim and the 1 s window for the Jolt_Filter are both centred on the frame and include the frame itself. The Still_Trim's windows take their frames from the whole Used_Part. The Jolt_Filter's windows take theirs from the Kept_Range only. A Stretch's length runs from its first frame's Frame_Time to the next frame's Frame_Time after the Stretch, or to the Used_Part's end.
- **A3 (Requirement 6):** The length that sets Target_Picks is Kept_Seconds, the length left after both the person's Trim and the Still_Trim. Target_Picks = min(`max_picks`, max(`base_picks`, ceil(Kept_Seconds / `seconds_per_pick`))), with `max_picks` = 400, `seconds_per_pick` = 2, and `base_picks` 120 for `fast`, 150 for `default` and 200 for `quality`.
- **A4 (Requirement 5):** "Fewer frames than we need" means the Jolt_Filter keeps fewer frames than Target_Picks. Then the step skips the Jolt_Filter for the whole clip and adds a warning to the Frames_Report.
- **A5 (Requirement 6):** The running total of Motion covers every frame of the Kept_Range, but only Candidates can become Picks. A Motion_Part with no Candidate gives no Pick, so the step gives at most Target_Picks Picks. When the Kept_Range's total Motion is 0, the step splits the Kept_Range by time. When two Candidates in a part are equally sharp, the earlier one is picked.
- **A6 (Requirements 1 and 3):** A Frame_Time is in seconds on the Trim's clock, where 0 s is the start of the video, and comes from the video's own timestamps, so a variable frame rate gets real times. The first Score_Frame's Motion is 0. Picks are unique and sorted by Frame_Time.
- **A7 (Requirement 2):** A video is an HDR_Clip when ffprobe reports the first video stream's colour transfer as `arib-std-b67` (HLG) or `smpte2084` (PQ). Only an HDR_Clip gets the Tone_Map. If ffmpeg has no `zscale` or `tonemap` filter, or fails with the Tone_Map, the step decodes without the Tone_Map and adds a warning. The Score_Pass makes that choice once, and the Pick_Pass follows it. `splatpipe/probe.py` doesn't read the colour transfer today. The design decides whether the Frames_Step runs its own ffprobe call or the Video_Probe gains the field.
- **A8 (Requirements 1, 3 and 8):** The step decodes the video twice. The Score_Pass gives raw grey Score_Frames, `score_width` (320) px wide, for scoring and for the Mask_Samples. The Pick_Pass gives the Picks in colour, with the long edge at `long_edge` (1600) px, and never scales a video up when it's smaller than that.
- **A9 (Requirement 8):** Each Grey_Copy is the Pick in grey with global histogram equalisation (OpenCV's `equalizeHist`), saved as a JPEG at `jpeg_quality` (95), with the same size and file name as its Colour_Pick, in a folder of its own. Only matching uses the Grey_Copies.
- **A10 (Requirement 8):** The Work_Folder holds `images/NNNNNN.jpg` (the Colour_Picks), `grey/NNNNNN.jpg` (the Grey_Copies), `frames.json` (the Frames_Record) and `mask.png` (the Mask, only when the step uses one). `NNNNNN` is the Pick's number in Frame_Time order, from `000000`, padded to 6 digits. These names stay in code as Fixed_Constants, because they're the contract with the Camera_Solve_Step. The step writes no files to `out/`.
- **A11 (Requirement 10):** A Step_Report's counts are whole numbers, so the seconds the Still_Trim cut at each end and the Mask_Coverage go in the Frames_Record. The Frames_Report counts frames instead, and gives whether the Mask is used as the count `mask_used`, 1 or 0.
- **A12 (Requirement 9), changed at review:** numpy and `opencv-python-headless` become runtime dependencies in `pyproject.toml`, whose `dependencies` is empty today, and their lines in the Licence_List change from "Not yet" to "Yes". The Frames_Step doesn't use scipy, so this spec doesn't add it, and scipy's line stays "Not yet". The clean-export spec, which plans scipy, decides whether to add it. The Skeleton check in `tests/test_imports.py` reads only `splatpipe/*.py`, so it doesn't stop `splatpipe/frames/` importing numpy and OpenCV.
- **A13 (Requirements 2 and 9), changed at review:** The fixed numbers are Shared_Settings in `[frames]`, and only `base_picks` is a Preset_Setting. They include the Tone_Map's two numbers: `tonemap_npl = 100`, the nominal peak brightness in nits that `zscale` maps from, and `tonemap_desat = 0.0`, how much `tonemap` desaturates bright highlights. The step checks them with the other settings and writes them into the Tone_Map's filters. The Tone_Map's `hable` curve stays a Fixed_Constant, because the user named it. The user confirmed `ffmpeg_timeout_seconds = 1800` for each Decode_Pass and `ffprobe_timeout_seconds = 30` at review. A Step sees only its own section, so `[frames]` has its own `ffprobe_timeout_seconds`, separate from the one in `[jobs]`.
- **A14 (Requirement 6):** `min_picks` is 30, confirmed by the user at review.
- **A15 (Requirements 1 and 7), changed at review:** The Mask_Samples come from the Sample_Buffer, which the Score_Pass fills in memory, with no temporary file. The Sample_Buffer keeps every k-th Score_Frame by Frame_Number, from the first with k = 1. Whenever it holds more than twice `mask_samples` (480) frames, it drops every other frame it holds and doubles k. So at the end it holds every Score_Frame when the Used_Part has at most 480, and otherwise 241 to 480 frames spread evenly by Frame_Number over the Used_Part. After the Still_Trim, the Mask_Samples are the Sample_Buffer's frames that lie inside the Kept_Range. When more than `mask_samples` (240) lie there, the step takes 240 of them, spread evenly among them with the first and last included. So the samples are spread evenly over the Used_Part and then limited to the Kept_Range: there are about 240 rather than exactly 240, fewer when the Still_Trim cut a large share of the Used_Part, and they aren't exactly even over the Kept_Range. At worst the Sample_Buffer holds 480 Score_Frames of 320 × 569 for a portrait 16:9 video, about 90 MB. With fewer than 2 Mask_Samples, the step uses no Mask.
- **A16 (Requirement 3):** The step works out the Contrast of every frame and records each Pick's Contrast in the Frames_Record. No rule in this step uses the Contrast.

### Open question, now resolved

- **O1 (Requirement 12, criterion 7):** ffmpeg's `zscale` filter comes from the zimg library (libzimg), which is under the WTFPL, version 2 ([zimg's own COPYING file](https://github.com/sekrit-twc/zimg/blob/master/COPYING)). The WTFPL isn't on the permissive list in `.kiro/steering/rules.md`. zimg is part of the ffmpeg command line tool on our machines, not our code or a Python or npm dependency, but the rules say to ask Samarth about any other licence. The review left this open: Samarth decides whether the Tone_Map can rely on zimg, and the task that builds the Tone_Map stops for his answer.
  - **Resolved in task 8.6:** Samarth OKed zimg, so the Tone_Map may rely on it. His answer was relayed in the chat during task 8, with his reasons: the WTFPL is permissive, and zimg comes only inside the ffmpeg command line tool on our private machine image, like ffmpeg's other libraries, so it isn't our code or a Python or npm dependency. The allowed list in `.kiro/steering/rules.md` stays as it is. Task 8.6 added zimg's row to the Licence_List, with "Not yet" for Installed, and its test in `tests/test_docs.py`.

Note on the clean-export spec: its A4 says the first spec with a Step worth running adds the first Step_List entry and changes the pipeline-core tests once, for everyone. This is that spec. If clean-export is built after this one, its Requirement 8, criterion 3 and the reasons it gives in Requirement 6, criterion 12 and Requirement 9, criterion 2 read as "adds no `clean_export` entry" rather than "leaves `STEP_LIST` empty".

## Glossary

- **Frames_Step**: the Step this spec builds, in `splatpipe/frames/`. Its Step_Name is `frames`, so its Config section is `[frames]`, its Work_Folder is `work/frames/` and its Step_Report is `out/reports/frames.json`.
- **Camera_Solve_Step**: the Step after this one, built in its own spec, which reads the Frames_Step's files from `work/frames/`.
- **ffmpeg**: the ffmpeg command line tool, which decodes the video. tech.md says every frame comes from it.
- **ffprobe**: the ffprobe command line tool, which comes with ffmpeg and reads what a video holds.
- **Decode_Pass**: one run of ffmpeg that reads the Used_Part and hands the decoded frames to the Frames_Step through a pipe.
- **Score_Pass**: the first Decode_Pass, which gives every Score_Frame.
- **Pick_Pass**: the second Decode_Pass, which gives the Picks in colour.
- **Score_Frame**: one frame of the Used_Part, decoded in grey, with brightness from 0 to 255, `score_width` px wide, and as tall as the video's shape needs.
- **Displayed_Shape**: the video's width and height as a player shows them, after the video's rotation is applied.
- **Frame_Time**: a frame's time in seconds from the start of the video, on the same clock as the Trim, taken from the video's own timestamps.
- **Frame_Number**: a Score_Frame's place among the Score_Frames of the Used_Part, from 0, in Frame_Time order.
- **HDR_Clip**: a video whose first video stream that isn't cover art has the colour transfer `arib-std-b67` (HLG) or `smpte2084` (PQ), as ffprobe reports it.
- **Tone_Map**: the ffmpeg filters that turn an HDR_Clip's frames into SDR BT.709: `zscale` with the nominal peak brightness `tonemap_npl`, and `tonemap` with the `hable` curve and the highlight desaturation `tonemap_desat` (A13).
- **Fixed_Constants**: what stays in code, not in Config, because it describes a format or a contract and not a choice we tune: the two HDR transfer names, the Tone_Map's filters with `hable` and BT.709 (its `npl` and `desat` numbers come from Config, A13), the Laplacian, the 0 to 255 brightness scale, the Mask's black 0 and white 255, and the Work_Folder's file and folder names (A10). This is the same exception the pipeline-core spec's Requirement 1, criterion 6 makes for the Job_Id pattern.
- **Sharpness**: the variance of the Laplacian of a Score_Frame.
- **Motion**: the mean, over every pixel, of the absolute change in brightness from the Score_Frame before, from 0 to 255. The first Score_Frame's Motion is 0.
- **Contrast**: the standard deviation of a Score_Frame's brightness.
- **Window**: for one frame and a length in seconds, the frames whose Frame_Time is at most half that length before or after the frame's own, the frame itself included.
- **Still_Frame**: a Score_Frame whose mean Motion over its Window of `motion_window_seconds` is below `still_motion`.
- **Stretch**: a run of Score_Frames that follow each other. Its length runs from its first frame's Frame_Time to the Frame_Time of the first frame after the Stretch, or to the Used_Part's end when no frame comes after it.
- **Blip**: a Stretch of frames that aren't Still_Frames, as long as it can be, lasting less than `blip_seconds` (A1).
- **Moving_Stretch**: a Stretch of frames that aren't Still_Frames, as long as it can be, lasting at least `blip_seconds`.
- **Still_Start**: the Stretch from the first Score_Frame up to the first Moving_Stretch, not including it.
- **Still_End**: the Stretch from just after the last Moving_Stretch to the last Score_Frame.
- **Still_Trim**: the part of the step that cuts the Still_Start and the Still_End off when each one is long enough (Requirement 4).
- **Kept_Range**: the Score_Frames the Still_Trim keeps, which are always one Stretch.
- **Kept_Seconds**: the Kept_Range's length, as a Stretch's length is measured.
- **Target_Picks**: the most Picks the step makes for a Job, N in the user's words (A3).
- **Jolt_Filter**: the part of the step that keeps only the Kept_Range frames that are sharp next to their neighbours (Requirement 5).
- **Candidates**: the frames that can become Picks: the frames the Jolt_Filter kept, or every frame of the Kept_Range when the step skips the Jolt_Filter.
- **Motion_Position**: a Kept_Range frame's running total of Motion, from the Kept_Range's first frame up to and including the frame, where the first frame's own Motion isn't counted.
- **Motion_Part**: one of the Target_Picks equal parts that the step splits the Kept_Range into, by Motion_Position, or by Frame_Time when the total Motion is 0.
- **Picker**: the part of the step that chooses the Picks from the Candidates (Requirement 6).
- **Pick**: a Score_Frame the Picker chose.
- **Pick_Size**: the Picks' width and height in pixels (Requirement 8, criterion 1).
- **Colour_Pick**: a Pick's colour JPEG in `work/frames/images/`.
- **Grey_Copy**: a Pick's grey, histogram-equalised JPEG in `work/frames/grey/`, used only for matching.
- **Sample_Buffer**: the Score_Frames the step keeps in memory during the Score_Pass for the Mask: at most twice `mask_samples` of them, thinned evenly by Frame_Number as the frames stream (Requirement 1, criterion 12, and A15).
- **Mask_Samples**: the Sample_Buffer's Score_Frames that the step uses to find the Stuck_Pixels (A15).
- **Pixel_Change**: for one pixel, the mean absolute change in its brightness from each Mask_Sample to the next.
- **Stuck_Pixel**: a pixel whose Pixel_Change is below `mask_ratio` times the median Pixel_Change of every pixel.
- **Mask_Coverage**: the share of a Score_Frame's pixels that are Stuck_Pixels, from 0 to 1.
- **Mask**: `work/frames/mask.png`, the black-and-white PNG that marks the Stuck_Pixels black, written only when the step uses it.
- **Frames_Record**: `work/frames/frames.json`, which describes the Picks and the Mask for the Camera_Solve_Step.
- **Frames_Report**: the Frames_Step's Step_Report, with the fields Requirement 10 lists.
- **Machine_Needs**: `min_vcpus` and `min_memory_gb` in `[frames]`, which the cloud spec reads to pick the machine.

Terms this spec uses as the pipeline-core spec and the Pipeline_Doc define them: Pipeline, Skeleton, Step, Step_Name, Job, Job_Id, Jobs_Folder, Job_Folder, Job_Record, Trim, Used_Part, Video_Length, Video_Probe, Step_Runner, Step_Report, Step_List, Required_Step, Optional_Step, Fake_Step, Work_Folder, Config, Preset, Shared_Setting, Preset_Setting, Step_Section, Expected_Failure, Unexpected_Error, Readable_Error, Job_Runner, Job_Status, JSON_Value, Pipeline_Doc, Licence_List, CLI and Test_Suite.

## Requirements

### Requirement 1: Reading the video

**User Story:** As someone who films a place, I want the step to read every frame of the part of my video I chose, so that the Picks come from that part and nothing else.

**Assumptions (confirmed at review):** A6 (criteria 5 and 6), A8 (criterion 4) and A15 (criterion 12). **Decided:** D4 (criteria 12 and 13) and D5 (criterion 14).

#### Acceptance Criteria

1. WHEN the Frames_Step runs, THE Frames_Step SHALL read the Job's copy of the video through `step.job.video_path` and the Used_Part from the Job_Record's `start_seconds` and `end_seconds`, and apply the Used_Part before any other part of the step's work.
2. THE Frames_Step SHALL take every decoded frame from ffmpeg's output and use OpenCV only on frames that ffmpeg has already decoded, as tech.md requires.
3. THE Frames_Step SHALL run ffmpeg and ffprobe as argument lists with no shell, with standard input closed, and with the video given as `file:` plus its absolute path, so a file name is never read as an option or a network address.
4. WHEN the Score_Pass runs, THE Frames_Step SHALL get one Score_Frame for each frame the video holds whose Frame_Time is at or after the Used_Part's start and before the Used_Part's end (A8).
5. THE Frames_Step SHALL decode the video's frames as the video holds them, with no frame repeated or dropped to keep a constant frame rate (A6).
6. THE Frames_Step SHALL take each frame's Frame_Time from the video's own timestamps, so a video with a variable frame rate gets the frames' real times, and the Frame_Times of the Score_Frames SHALL increase strictly from one Score_Frame to the next (A6).
7. THE Frames_Step SHALL decode the Score_Pass and the Pick_Pass with the same rotation and the same choice about the Tone_Map, so each Pick shows the same view as its Score_Frame, larger and in colour.
8. IF ffmpeg or ffprobe isn't installed, THEN THE Frames_Step SHALL fail with an Expected_Failure whose Readable_Error names the missing tool and says to install ffmpeg, which includes ffprobe.
9. IF a Decode_Pass runs past `ffmpeg_timeout_seconds`, or ffprobe runs past `ffprobe_timeout_seconds`, THEN THE Frames_Step SHALL stop the tool and fail with an Expected_Failure whose Readable_Error names the limit and the setting in `[frames]` that sets the limit.
10. IF ffmpeg or ffprobe exits with a non-zero code, other than in the Tone_Map case of Requirement 2, criterion 5, THEN THE Frames_Step SHALL fail with an Expected_Failure whose Readable_Error holds the tool's last line of error output, with no traceback.
11. IF the Score_Pass gives no Score_Frame inside the Used_Part, THEN THE Frames_Step SHALL fail with an Expected_Failure whose Readable_Error says that no frames could be read from the chosen part of the video and says to check that the video plays.
12. WHILE a Decode_Pass runs, THE Frames_Step SHALL take the frames one at a time from ffmpeg's pipe and hold, whatever the video's length, at most the Score_Frame before, the current Score_Frame and the Sample_Buffer's at most twice `mask_samples` Score_Frames in the Score_Pass, and one colour frame at a time in the Pick_Pass, keeping only each frame's Frame_Time and scores for every frame. The Sample_Buffer SHALL keep every k-th Score_Frame by Frame_Number, from the first with k = 1, and whenever it holds more than twice `mask_samples` Score_Frames, drop every other Score_Frame it holds and double k (D4, A15).
13. THE Frames_Step SHALL keep its peak memory, ffmpeg included, under 16 GB for a 10-minute 4K clip, which is the design target behind criterion 12 and the Machine_Needs (D4).
14. IF ffmpeg's output can't be read, because a frame comes without its time or the output ends inside a frame, THEN THE Frames_Step SHALL fail with an Expected_Failure whose Readable_Error says that ffmpeg's output couldn't be read and says to install ffmpeg 5.1 or newer (D5).

Note for the design: ffmpeg can give two frames the same timestamp. The design says how the step keeps the Frame_Times strictly increasing (criterion 6) and how it reads each frame's timestamp alongside the raw frames.

### Requirement 2: HDR video

**User Story:** As someone who films a place on a phone that shoots HDR, I want my video's colours turned into normal SDR before anything is scored, so that the Picks don't look grey and washed out.

**Assumptions (confirmed at review):** A7 (criteria 1 to 6) and A13 (criterion 2).

#### Acceptance Criteria

1. WHEN the Frames_Step starts the step's work, THE Frames_Step SHALL ask ffprobe for the colour transfer of the video's first video stream that isn't cover art (A7).
2. WHERE the video is an HDR_Clip, THE Frames_Step SHALL run the Tone_Map, with `tonemap_npl` and `tonemap_desat` from Config, on each frame in both Decode_Passes, before the frame is turned grey or saved as a Pick (A7, A13).
3. WHERE the video isn't an HDR_Clip, including a video whose colour transfer ffprobe doesn't report, THE Frames_Step SHALL decode the video without the Tone_Map (A7).
4. IF the video is an HDR_Clip and ffmpeg has no `zscale` filter or no `tonemap` filter, THEN THE Frames_Step SHALL decode the video without the Tone_Map and add a warning to the Frames_Report that names the colour transfer, says the Picks' colours may look flat, and says that an ffmpeg with `zscale` fixes the colours (A7).
5. IF the video is an HDR_Clip and the Score_Pass with the Tone_Map exits with a non-zero code, THEN THE Frames_Step SHALL run the Score_Pass again without the Tone_Map and add the warning of criterion 4, with ffmpeg's last line of error output (A7).
6. THE Frames_Step SHALL choose whether to use the Tone_Map once, in the Score_Pass, and SHALL run the Pick_Pass with the same choice (A7).
7. THE Frames_Record SHALL hold the colour transfer ffprobe reported, or `null` when ffprobe reported none, and whether the Tone_Map ran.

### Requirement 3: Scoring every frame

**User Story:** As a team member, I want every frame scored for sharpness, motion and contrast, so that the rules that cut, filter and pick frames have numbers to work on.

**Assumptions (confirmed at review):** A6 (criterion 4), A8 (criterion 1) and A16 (criterion 6).

#### Acceptance Criteria

1. THE Frames_Step SHALL make each Score_Frame grey, with brightness from 0 to 255, exactly `score_width` px wide, and with the height that keeps the Displayed_Shape, rounded to a whole number of pixels (A8).
2. THE Frames_Step SHALL give each Score_Frame a Sharpness: the variance of the Score_Frame's Laplacian.
3. THE Frames_Step SHALL give each Score_Frame a Motion: the mean absolute change in brightness from the Score_Frame before, over every pixel.
4. THE Frames_Step SHALL give the first Score_Frame of the Used_Part a Motion of 0 (A6).
5. THE Frames_Step SHALL give each Score_Frame a Contrast: the standard deviation of the Score_Frame's brightness.
6. THE Frames_Step SHALL keep each Score_Frame's Frame_Number, Frame_Time, Sharpness, Motion and Contrast, and record each Pick's Contrast in the Frames_Record. No rule in the Frames_Step uses the Contrast (A16).
7. FOR ALL Score_Frames, THE Frames_Step SHALL give a Motion from 0 to 255, a Sharpness of 0 or more and a Contrast from 0 to 127.5 (property).

### Requirement 4: Trimming the still ends

**User Story:** As someone who films a place, I want the seconds where I held the phone still before and after walking cut off, so that the Picks aren't wasted on one view.

**Assumptions (confirmed at review):** A1 (criteria 2 and 3) and A2 (criteria 1 and 4).

#### Acceptance Criteria

1. THE Still_Trim SHALL treat a Score_Frame as a Still_Frame when the mean Motion over the frame's Window of `motion_window_seconds`, taken from the frames of the Used_Part, is below `still_motion` (A2).
2. THE Still_Trim SHALL count each Blip as still, so a Blip inside a still Stretch doesn't end the still Stretch (A1).
3. THE Still_Trim SHALL find the Still_Start and the Still_End from the Moving_Stretches, which last at least `blip_seconds` (A1).
4. WHEN the Still_Start lasts `min_still_seconds` or more, THE Still_Trim SHALL cut the Still_Start's frames off the Kept_Range (A2).
5. WHEN the Still_End lasts `min_still_seconds` or more, THE Still_Trim SHALL cut the Still_End's frames off the Kept_Range (A2).
6. WHERE the Still_Start or the Still_End lasts less than `min_still_seconds`, THE Still_Trim SHALL keep that Stretch's frames in the Kept_Range.
7. IF the Used_Part has no Moving_Stretch, THEN THE Still_Trim SHALL keep every Score_Frame in the Kept_Range, so a clip that is still from start to end is kept whole (D2).
8. THE Still_Trim SHALL cut frames only at the start and the end, and keep each still Stretch that lies between two Moving_Stretches.
9. FOR ALL lists of at least one Score_Frame, with strictly increasing Frame_Times and any Motions from 0 to 255, THE Still_Trim SHALL give a Kept_Range that holds at least one frame and is one Stretch, and SHALL keep every frame when the list has no Moving_Stretch (property).
10. THE Frames_Record SHALL hold the seconds the Still_Trim cut at the start and at the end, 0 when the Still_Trim cut nothing there, and the Kept_Range's start and end in seconds (A11).

### Requirement 5: Dropping jolts

**User Story:** As someone who films a place, I want the frames blurred by a jolt left out, so that the splat isn't built from smeared pictures.

**Assumptions (confirmed at review):** A2 (criterion 1) and A4 (criteria 2 and 3).

#### Acceptance Criteria

1. THE Jolt_Filter SHALL keep a Kept_Range frame when the frame's Sharpness is at least `jolt_ratio` times the median Sharpness over the frame's Window of `jolt_window_seconds`, taken from the frames of the Kept_Range (A2).
2. IF the Jolt_Filter keeps fewer frames than Target_Picks, THEN THE Frames_Step SHALL skip the Jolt_Filter for the whole clip and make every frame of the Kept_Range a Candidate (A4).
3. WHEN the Frames_Step skips the Jolt_Filter, THE Frames_Step SHALL add a warning to the Frames_Report that gives how many frames the Jolt_Filter kept and the Target_Picks, and says the Jolt_Filter was skipped (A4).
4. WHEN the Jolt_Filter keeps at least Target_Picks frames, THE Frames_Step SHALL make exactly the frames the Jolt_Filter kept the Candidates.
5. FOR ALL Kept_Ranges, with any Frame_Times that strictly increase and any Sharpness values of 0 or more, every frame the Jolt_Filter keeps SHALL meet the rule of criterion 1 and every frame the Jolt_Filter drops SHALL break the rule (property).

### Requirement 6: Spreading the picks over the camera's movement

**User Story:** As someone who films a place, I want the Picks spread evenly over how far the camera moved, so that a slow part of my walk doesn't get more Picks than a fast part.

**Assumptions (confirmed at review):** A3 (criteria 1 and 2), A5 (criteria 3 to 7) and A14 (criterion 9). **Decided:** D2 (criteria 9 and 10).

#### Acceptance Criteria

1. THE Picker SHALL set Target_Picks to min(`max_picks`, max(`base_picks`, ceil(Kept_Seconds / `seconds_per_pick`))), where `base_picks` is the Preset_Setting for the Job's Preset (A3).
2. THE Picker SHALL give a Target_Picks of 120 for a Kept_Seconds of 240 s with the `fast` Preset, and 210 for a Kept_Seconds of 420 s with the `fast` Preset, with the values of Requirement 9 (A3).
3. WHEN the Kept_Range's total Motion, the last frame's Motion_Position, is above 0, THE Picker SHALL split the Motion from 0 to the total into Target_Picks equal Motion_Parts, and put each Kept_Range frame in the Motion_Part that holds the frame's Motion_Position, with the last Motion_Part holding the total (A5).
4. WHEN the Kept_Range's total Motion is 0, THE Picker SHALL split the Kept_Range's time, from the first frame's Frame_Time to the Kept_Range's end, into Target_Picks equal Motion_Parts, and put each Kept_Range frame in the Motion_Part that holds the frame's Frame_Time (A5).
5. THE Picker SHALL pick, from each Motion_Part, the Candidate with the highest Sharpness, and the earliest of those when two or more have the same Sharpness (A5).
6. WHERE a Motion_Part holds no Candidate, THE Picker SHALL make no Pick for that Motion_Part (A5).
7. THE Picker SHALL give the Picks in Frame_Time order (A5).
8. FOR ALL Kept_Ranges, Candidates, Motions, Sharpness values and Target_Picks, THE Picker SHALL give at most Target_Picks Picks, each a different Candidate inside the Kept_Range and the Used_Part, with Frame_Times that strictly increase (property).
9. IF the Picker gives fewer Picks than `min_picks`, THEN THE Frames_Step SHALL fail with an Expected_Failure before the Pick_Pass runs, and write no Colour_Pick, Grey_Copy, Mask or Frames_Record (D2, A14).
10. THE Readable_Error for criterion 9 SHALL give how many Picks the video gave and how many the step needs, and say what to film differently: a longer video that moves slowly and steadily through the place, in good light, without long still parts (D2).

### Requirement 7: Masking what is stuck to the camera

**User Story:** As someone who films with a drone or an app that adds a watermark, I want the parts of the picture that never change left out, so that the camera solve and the training don't treat them as part of the place.

**Assumptions (confirmed at review):** A15 (criteria 1 and 7). **Decided:** D3 (criteria 1 to 10).

#### Acceptance Criteria

1. THE Frames_Step SHALL take as Mask_Samples the Sample_Buffer's Score_Frames that lie inside the Kept_Range, and, when more than `mask_samples` of them lie there, only `mask_samples` of them, spread evenly among them with the first and last included (D3, A15).
2. THE Frames_Step SHALL work out each pixel's Pixel_Change from the Mask_Samples in Frame_Time order (D3).
3. THE Frames_Step SHALL mark as a Stuck_Pixel each pixel whose Pixel_Change is below `mask_ratio` times the median Pixel_Change of every pixel (D3).
4. WHEN the Mask_Coverage is at least `mask_min_coverage` and at most `mask_max_coverage`, THE Frames_Step SHALL use the Mask and write the Mask (D3).
5. IF the Mask_Coverage is below `mask_min_coverage` or above `mask_max_coverage`, THEN THE Frames_Step SHALL use no Mask and write no Mask file (D3).
6. THE Frames_Step SHALL use the Stuck_Pixels as found, without growing, shrinking or smoothing them (D3).
7. IF there are fewer than 2 Mask_Samples, THEN THE Frames_Step SHALL use no Mask and write no Mask file (A15).
8. THE Mask SHALL be one 8-bit grey PNG at the Pick_Size, holding only the values 0 and 255, with each Stuck_Pixel black (0, ignore) and every other pixel white (255, use), scaled up from the Score_Frame's size so that each pixel takes the value of the Score_Frame pixel it covers (D3).
9. THE Frames_Record SHALL hold the Mask_Coverage and whether the step uses the Mask, with the Mask's file name when the step uses the Mask and `null` when the step doesn't (D3).
10. THE Mask SHALL suit COLMAP's single camera mask, `--ImageReader.camera_mask_path`, which ignores black pixels, and the training loss, which reads the same file (D3).
11. FOR ALL lists of Mask_Samples, THE Frames_Step SHALL either use no Mask or use a Mask whose Mask_Coverage is at least `mask_min_coverage` and at most `mask_max_coverage` (property).

### Requirement 8: Writing the picks

**User Story:** As a team member building the camera solve, I want the Picks, their grey copies, their times and the Mask in one known layout, so that my step can read them without guessing.

**Assumptions (confirmed at review):** A8 (criterion 1), A9 (criterion 3) and A10 (criteria 4 to 8).

#### Acceptance Criteria

1. THE Frames_Step SHALL set the Pick_Size to the Displayed_Shape scaled so that the long edge is `long_edge` px when the Displayed_Shape's long edge is longer than `long_edge`, and to the Displayed_Shape itself otherwise, so a Pick is never larger than the video (A8).
2. WHEN the Pick_Pass runs, THE Frames_Step SHALL save each Pick as a Colour_Pick: a JPEG at `jpeg_quality` and the Pick_Size, made from the same video frame as the Pick's Score_Frame.
3. WHEN the Frames_Step saves a Colour_Pick, THE Frames_Step SHALL also save the Pick's Grey_Copy: the same decoded frame turned grey with global histogram equalisation, saved as a JPEG at `jpeg_quality` with the same size and file name as the Colour_Pick (A9).
4. THE Frames_Step SHALL save the Colour_Picks as `work/frames/images/NNNNNN.jpg` and the Grey_Copies as `work/frames/grey/NNNNNN.jpg`, where `NNNNNN` is the Pick's number in Frame_Time order, from `000000`, padded to 6 digits (A10).
5. THE Frames_Step SHALL write the Frames_Record to `work/frames/frames.json` and the Mask, when the step uses one, to `work/frames/mask.png` (A10).
6. THE Frames_Record SHALL hold only JSON_Values: what Requirement 2, criterion 7, Requirement 4, criterion 10 and Requirement 7, criterion 9 list, the Target_Picks, whether the step skipped the Jolt_Filter, the Pick_Size, and for each Pick its number, its file name, its Frame_Number, its Frame_Time in seconds, its Sharpness, its Motion and its Contrast (A10, A11).
7. THE Frames_Step SHALL write the Frames_Record after every Colour_Pick, every Grey_Copy and the Mask, so a run that fails leaves no Frames_Record, and the Camera_Solve_Step can treat a Work_Folder without a Frames_Record as unfinished (A10).
8. THE Frames_Step SHALL write the step's files only inside the Work_Folder and write no files to `out/`, where only the Step_Runner writes the Frames_Report (A10).
9. IF the Pick_Pass ends before giving every Pick, THEN THE Frames_Step SHALL fail with an Expected_Failure whose Readable_Error says how many Picks ffmpeg gave back out of how many were chosen.
10. IF writing a Colour_Pick, a Grey_Copy, the Mask or the Frames_Record fails because the disk is full or the file can't be written, THEN THE Frames_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and the reason.

### Requirement 9: Settings, presets and dependencies

**User Story:** As a team member, I want every number of this step in Config, so that we can tune the frame picking without changing code.

**Assumptions (confirmed at review):** A1 (criterion 2), A3 (criterion 3), A12 (criteria 8 and 9), A13 (criteria 2, 3, 4 and 6) and A14 (criterion 2). **Decided:** D4 (criteria 2 and 10).

#### Acceptance Criteria

1. THE Config SHALL hold a `[frames]` Step_Section with the three Preset tables `[frames.fast]`, `[frames.default]` and `[frames.quality]`, each with the same keys, and with a short comment for every key.
2. THE `[frames]` section SHALL hold these Shared_Settings: `score_width = 320`, `tonemap_npl = 100`, `tonemap_desat = 0.0`, `still_motion = 1.0`, `motion_window_seconds = 1`, `min_still_seconds = 2`, `blip_seconds = 0.5`, `jolt_ratio = 0.6`, `jolt_window_seconds = 1`, `mask_samples = 240`, `mask_ratio = 0.2`, `mask_min_coverage = 0.002`, `mask_max_coverage = 0.25`, `jpeg_quality = 95`, `long_edge = 1600`, `seconds_per_pick = 2`, `max_picks = 400`, `min_picks = 30`, `min_vcpus = 16`, `min_memory_gb = 16`, `ffmpeg_timeout_seconds = 1800` and `ffprobe_timeout_seconds = 30` (A1, A13, A14, D4).
3. THE Preset tables SHALL each hold one Preset_Setting, `base_picks`: 120 in `[frames.fast]`, 150 in `[frames.default]` and 200 in `[frames.quality]` (A3).
4. THE Frames_Step SHALL take every threshold, count, size, ratio and time limit from Config, and hold in code only the Fixed_Constants.
5. IF a setting is missing from `[frames]` and its Preset table, THEN THE Step_Runner's existing check SHALL fail the Step with the Step_Runner's Readable_Error, which names the section, the table and the key.
6. IF a setting is present but isn't the kind of value the setting must be, THEN THE Frames_Step SHALL fail with an Expected_Failure, before ffprobe or ffmpeg runs, whose Readable_Error names the setting, the section and what the setting must be: `score_width`, `long_edge`, `max_picks`, `min_picks`, `base_picks`, `min_vcpus` and `min_memory_gb` whole numbers above 0, `mask_samples` a whole number of 2 or more, `jpeg_quality` a whole number from 1 to 100, `still_motion`, `motion_window_seconds`, `min_still_seconds`, `blip_seconds`, `jolt_window_seconds`, `seconds_per_pick`, `tonemap_npl`, `ffmpeg_timeout_seconds` and `ffprobe_timeout_seconds` numbers above 0, `tonemap_desat` a number of 0 or more (0 turns the `tonemap` filter's highlight desaturation off), `jolt_ratio` and `mask_ratio` numbers from 0 to 1, and `mask_min_coverage` and `mask_max_coverage` numbers from 0 to 1 with `mask_min_coverage` no greater than `mask_max_coverage`. A bool is never a number here.
7. IF `min_picks` is greater than `base_picks`, or `base_picks` is greater than `max_picks`, THEN THE Frames_Step SHALL fail with an Expected_Failure, before ffprobe or ffmpeg runs, whose Readable_Error names the settings and their values.
8. THE `pyproject.toml` SHALL gain only numpy and `opencv-python-headless` as runtime dependencies: no scipy, never plain `opencv-python` and no GPU package, so `uv sync` on a laptop still installs no torch, no gsplat and no CUDA wheel (A12).
9. THE Frames_Step SHALL import only the Python standard library, numpy, OpenCV (`cv2`) and `splatpipe`, and import nothing from `cloud/`, boto3, torch or gsplat.
10. THE Frames_Step SHALL run the same whatever the machine's CPUs and memory, and leave the Machine_Needs for the cloud spec to read (D4).
11. THE `tests/conftest.py` test config SHALL gain a `[frames]` section with the section's three Preset tables, so a test that runs the step through the step's CLI or `run()` without a Config of the test's own gets settings, and a test SHALL be able to change any one of the settings for that test.

### Requirement 10: The step report

**User Story:** As a team member, I want the step's report to say how many frames it read, cut, dropped and picked, so that I can tell a good run from one that threw the walk away.

**Assumptions (confirmed at review):** A11 (criteria 1 and 4).

#### Acceptance Criteria

1. WHEN the Frames_Step is done, THE Frames_Report SHALL hold the counts `decoded_frames` (Score_Frames in the Used_Part), `still_start_frames` and `still_end_frames` (frames the Still_Trim cut at each end), `kept_frames` (frames in the Kept_Range), `jolt_dropped_frames` (frames the Jolt_Filter dropped, 0 when the step skipped the Jolt_Filter), `target_picks`, `picks`, `mask_samples`, `mask_pixels` (Stuck_Pixels at the Score_Frame's size), `mask_used` (1 when the step uses the Mask, 0 when the step doesn't), `pick_width` and `pick_height` (A11).
2. WHEN the Frames_Step is done, THE Frames_Report SHALL hold the timings `probe`, `score_pass`, `choose` (the Still_Trim, the Jolt_Filter, the Picker and the Mask) and `pick_pass`, each in seconds.
3. WHEN the Frames_Step fails, THE Frames_Report SHALL hold exactly the counts and timings of criteria 1 and 2 that the step had worked out before the step failed.
4. THE Frames_Step SHALL add each count as a whole number given as an int and each timing as a number of seconds, so the Step_Runner's checks never refuse one, and SHALL keep the fractional seconds and the Mask_Coverage in the Frames_Record (A11).
5. WHERE the step warns under Requirement 2, criteria 4 and 5, or Requirement 5, criterion 3, THE Frames_Report SHALL hold that warning.
6. THE Frames_Report SHALL be the Step_Runner's ordinary Step_Report, written to `out/reports/frames.json`, with the settings of Requirement 9, criteria 2 and 3, which the Step_Runner puts there itself.

### Requirement 11: Running the step and the step list

**User Story:** As a team member, I want the Frames_Step to run on its own and as the first step of every Job, so that `splatpipe.run` does real work on a real Job.

**Decided:** D1 (criteria 3 to 9).

#### Acceptance Criteria

1. THE Frames_Step SHALL be a package `splatpipe/frames/` built like the Fake_Step: `work(step)`, `run(job_id, preset=None, *, config=None)` and a `__main__.py` whose `main(argv)` calls the Skeleton's `step_main`.
2. THE CLI `uv run python -m splatpipe.frames --job <id> [--preset P]` SHALL run the step on one Job, print `frames: done` on standard output and exit with 0 when the step is done, and print the Readable_Error on standard error with no traceback and exit with 1 when the step failed.
3. THE Step_List in `splatpipe/run.py` SHALL be `(StepEntry("frames", "splatpipe.frames", True),)`, with the Step_List's comment saying that the Frames_Step is the first entry and that each later step spec adds the spec's own entry after it, in order (D1).

   Note: The cameras spec added the second Step_List entry, `StepEntry("cameras", "splatpipe.cameras", True)`. This criterion describes the Step_List as the frames spec left it.

4. WHILE the Frames_Step is the only entry in the Step_List, THE Job_Runner SHALL give a Job the Job_Status `done` once the Frames_Step is done (D1).

   Note: The cameras spec added the second Step_List entry, `StepEntry("cameras", "splatpipe.cameras", True)`. This criterion describes the Step_List as the frames spec left it.

5. IF a caller gives the Job_Runner an empty Step_List, THEN THE Job_Runner SHALL still refuse with the Readable_Error that starts "No steps are set up yet", before the Job_Runner writes anything (D1).
6. THE `tests/test_run.py` SHALL change in the same task as the Step_List: the test that asserted `STEP_LIST == ()` SHALL assert the Step_List of criterion 3, the test that ran the default Step_List with `steps=None` and the CLI test that expected "No steps are set up yet" SHALL set the Step_List to empty for that test, and the test that runs `python -m splatpipe.run --job ../x` SHALL expect "That isn't a job id." (D1).
7. THE `tests/test_imports.py` SHALL load `[frames]` from the real Config for each of the three Presets through the existing Step_List check, with the check's docstrings no longer saying the Step_List starts empty (D1).
8. THE `tests/test_docs.py` SHALL find a `## Step: frames` section in the Pipeline_Doc through the existing Step_List check (D1).
9. THE pipeline-core spec's requirements document SHALL gain a note under Requirement 8, criterion 2 saying that the frames spec added the first Step_List entry, and leave the criterion's own words unchanged (D1).
10. THE Frames_Step SHALL raise `ExpectedFailure` for every failure that Requirements 1, 2, 6, 8 and 9 name, so the Frames_Report's `error.expected` is `true` and the report's `readable` is the step's own message.
11. WHEN the Frames_Step runs again on a Job, THE Frames_Step SHALL rely on the Step_Runner emptying `work/frames/` first, so every file of an earlier run is gone before the new run writes.

### Requirement 12: Docs and licences

**User Story:** As one of the three people building the steps, I want this step written up next to the others, so that the camera solve and the cloud wrapper fit with it.

**Decided:** D1 (criterion 5), D3 (criterion 3), D4 (criterion 4) and D5 (criterion 8). **Open question, now resolved (Samarth OKed zimg):** O1 (criterion 7).

#### Acceptance Criteria

1. THE Pipeline_Doc SHALL gain a `## Step: frames` section with the same parts as the Fake_Step's section: a `### What it reads`, a `### What it writes`, a `### Config section`, a `### Step report fields` and a `### Files in out/` part.
2. THE `### Files in out/` part SHALL say that the Frames_Step writes no files to `out/` other than the Frames_Report, which the Step_Runner writes.
3. THE `### What it writes` part SHALL give the Work_Folder's layout and the Frames_Record's fields as the contract the Camera_Solve_Step reads, say that only matching uses the Grey_Copies, and say that COLMAP takes the Mask as `--ImageReader.camera_mask_path` and the training loss reads the same file (D3).
4. THE Frames_Step's section SHALL state the Machine_Needs, at least 16 vCPUs and 16 GB of memory on the Job's cloud machine, and say that the step itself never checks the machine (D4).
5. THE Pipeline_Doc's "The step list" section SHALL say that the Step_List holds the Frames_Step as the Step_List's first, required entry, that each later step spec adds the spec's own entry after it, and that a Job is done after the Frames_Step alone until the camera solve spec adds the camera solve's entry, in place of the paragraph that says the list starts empty (D1).

   Note: The cameras spec added the second Step_List entry, `StepEntry("cameras", "splatpipe.cameras", True)`. This criterion describes the Step_List as the frames spec left it.

6. THE Licence_List SHALL mark numpy and `opencv-python-headless` as installed and keep scipy's row at "Not yet", and the ffmpeg row SHALL say that ffmpeg decodes the frames of the Used_Part and tone maps HDR video with `zscale` and `tonemap` when the ffmpeg build has both filters, in the same task that adds the dependencies.
7. THE Licence_List SHALL gain a line for zimg (libzimg, WTFPL), which ffmpeg's `zscale` filter needs, only after Samarth has OKed the licence, and the task that builds the Tone_Map SHALL stop for Samarth's answer, because the review left O1 open (O1).
8. THE Frames_Step's section SHALL say that the Job's machine image needs ffmpeg and ffprobe 5.1 or newer, because the Frames_Step uses ffmpeg's `-fps_mode` option, and that Ubuntu 24.04's packaged ffmpeg, version 6.1, is new enough (D5).

### Requirement 13: Tests

**User Story:** As a team member, I want the step's maths tested on a laptop CPU with tiny made-up frames, so that the tests are quick, free and safe to run anywhere.

#### Acceptance Criteria

1. THE Test_Suite SHALL test the Sharpness, the Motion, the Contrast, the Still_Trim (Blips and still ends shorter than `min_still_seconds` included), the Jolt_Filter (the skip included), the Target_Picks (the examples of Requirement 6, criterion 2 included), the Picker (a total Motion of 0, empty Motion_Parts and equal Sharpness included), the Sample_Buffer's thinning, the choice of Mask_Samples from the Sample_Buffer, the Stuck_Pixels and the Mask's use, the setting checks and the report fields with unit tests on tiny synthetic numpy frames the tests build themselves.
2. THE Test_Suite SHALL hold the property tests of Requirement 3, criterion 7, Requirement 4, criterion 9, Requirement 5, criterion 5, Requirement 6, criterion 8 and Requirement 7, criterion 11, written with Hypothesis.
3. THE Test_Suite's unit tests and property tests SHALL both be required, never optional.
4. THE Test_Suite SHALL test the Decode_Passes on tiny clips that ffmpeg makes in a temporary folder: that the Used_Part is respected, that each Colour_Pick comes from the same frame as the Pick's Score_Frame, that a clip with a variable frame rate gets the frames' real times, that a video smaller than `long_edge` isn't scaled up, and that the Work_Folder's layout matches Requirement 8.
5. IF ffmpeg or ffprobe isn't installed, THEN THE Test_Suite SHALL skip the tests that need ffmpeg or ffprobe and say why, and IF ffmpeg has no `zscale` filter, THEN THE Test_Suite SHALL skip the test that runs the real Tone_Map and say why.
6. THE Test_Suite SHALL test the Tone_Map fallback of Requirement 2, criteria 4 and 5, and the ffmpeg and ffprobe failures of Requirement 1, criteria 8 to 10, with a stand-in for the tools, so the whole suite passes on a machine with no ffmpeg.
7. THE Test_Suite SHALL check criterion 12 of Requirement 1 by feeding the step's scoring a long stream of tiny synthetic frames and checking how many frames the step holds at once, and SHALL check that the Sample_Buffer never holds more than twice `mask_samples` Score_Frames, holds every frame of a stream of at most twice `mask_samples` frames, and otherwise holds every k-th frame by Frame_Number from 0.
8. THE Test_Suite SHALL check that a run that fails, with too few Picks and in the Pick_Pass, leaves no Frames_Record (Requirement 6, criterion 9 and Requirement 8, criterion 7).
9. THE Test_Suite SHALL check that each Readable_Error of the step's Expected_Failures names what the requirement says the Readable_Error names and holds no traceback.
10. THE Test_Suite SHALL check that no file in `splatpipe/frames/` imports anything outside the standard library, numpy, `cv2` and `splatpipe`, so torch, gsplat, boto3 and `cloud/` stay out (Requirement 9, criterion 9).
11. THE Test_Suite SHALL run on the CPU with no network, make every Job in a temporary Jobs_Folder and never in the real `data/` folder, commit no video or frame to git, and give every subprocess a timeout.
