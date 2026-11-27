# Requirements Document

## Introduction

The Website is the web app around the Pipeline. Judges, and later other users, open it in a browser on a laptop or a phone. Someone signs in with an Account that the Account_Owner made, uploads a phone video of a place, picks a Preset (fast unless they choose another) and starts a Job. The Website lists that Account's Jobs with each one's status, read from the Job's files in the Job_Bucket. When a Job is done, its Result_Page shows the result and offers one download: `out/clean.sog`, or `out/train.ply` when the Job has no clean copy. A Showcase of finished Jobs that the Account_Owner picks lets a judge see results without waiting.

The Website starts each Job through the same code as `uv run python -m cloud.start`. This spec splits the Start_Command's `start_job` into a shared Start_Core and two thin callers, so the Smoke_Marker gate, the lock-hash check, the G_Quota check, the Spend_Guard and the fallback launch apply to every Website_Job. When every GPU type is full, the Website says so and asks the person to try again later. Nothing waits in a queue.

Every Job costs money, so the Website enforces three Website_Limits set in Config: 10 Website_Jobs a UTC_Day, $52 a UTC_Day and $500 in total, until the Stop_Time of 08:00 UTC on 4 February 2027. Each start takes a Reservation of the Job's Worst_Case_Cost in one atomic write. The Account_Owner's `APPROVED_SPEND=1` on the Deploy_Command is the one yes for those limits, in place of a yes per Job.

The Website runs as one Lambda function with a function URL, which gives HTTPS with no domain to buy, and one on-demand DynamoDB table, so it costs about nothing while nobody uses it. Big files never pass through the function: the browser sends the video straight to the Job_Bucket in parts, and each download is a short-lived link to one file. No AWS secret key and no other secret reaches the browser. The Website's code lives in `app/`, with its own App_Project that holds only boto3, so the root `uv.lock`, which every Job's Machine_Image must match, never changes for the Website.

Deploying is the Account_Owner's job, so this spec ends with the Deploy_Command and its docs, not with a deploy. The Deploy_Command, the Account_Command and the Showcase_Command run only in CloudShell with the Admin_Login, and Kiro never runs them.

### In scope

- The Website in `app/`: the request handler, the pages, the Page_Script, signing in and out, Sessions, CSRF_Tokens and the sign-in limit, the Upload, the Video_Check and its Box_Reader, starting a Job, the Website_Limits and Reservations, the job list, the Website_Statuses, the Result_Page with its Viewer_Slot, the Download_Links and the Showcase page.
- The App_Project, `app/pyproject.toml` and `app/uv.lock`, with boto3 as its only dependency.
- The Start_Core split in `cloud/start.py`, with the Laptop_Caller and the Website_Caller.
- The Website's new AWS operations on the `Aws` seam in `cloud/aws.py` (DynamoDB reads and conditional writes, S3 multipart uploads, Presigned_Links, copies inside S3, object sizes and range reads), and the same operations in the Fake_AWS in `tests/fake_aws.py`.
- The `[website]` Config section.
- The Stack_Template's additions: the Website_Function, its Website_URL and its log group, the Website_Table, the Website_Role, one CORS rule and one lifecycle rule on the Job_Bucket, the Website_Code_Parameters and the Placeholder_Function.
- The three CloudShell_Commands: the Deploy_Command, the Account_Command and the Showcase_Command.
- The Guard_Hook's block of the CloudShell_Commands' modules on a laptop, with its cases in `tests/test_guard.py` (R9).
- The updates to the Setup_Guide, the Rules_Steering, the Tech_Steering, the Structure_Steering, the Product_Steering, the Cloud_Steering, the AWS_Jobs_Skill, the Licence_List, the README and the Pipeline_Doc.
- The tests, all against the Fake_AWS.

### Out of scope

- The 3D viewer and the splat-transform version it's pinned to. Both belong to the Viewer_Spec. This spec leaves the Viewer_Slot empty.
- Deploying the Website or updating the Stack, running a CloudShell_Command, and any AWS call or paid run while carrying out this spec.
- A sign-up page, a password reset by email, or any way to make an Account other than the Account_Command.
- A queue of Jobs, a Trim on the Website, a custom domain, and API Gateway, CloudFront, App Runner or an EC2 web server.
- Any change to `splatpipe/`, to the Project_File or to the root `uv.lock`, and any Step_List entry.
- Any change to the rights of the Read_Profile or the Runner_Profile.

### Task budget

The whole spec stays within 10 tasks, which the design and tasks phases enforce. Each task writes its own unit tests and property tests, both required, never optional.

### Settled in clarification

The clarify step asked two rounds of questions. The user answered round 1 in their own words. For round 2 the user took the recommended answers ("the rest as recommended") and changed only Q11. Each answer is marked in the requirement it belongs to.

Round 1, in the user's words:

- **Q1 — Passwords (Requirement 3, Requirement 4):** "hash passwords with hashlib.scrypt from the standard library, and the account command makes long random passwords." The Account_Command shows a new password once, to the Account_Owner.
- **Q2 — The Stop_Time (Requirement 2, Requirement 10, Requirement 11, Requirement 16):** "stop starting jobs at 08:00 UTC on 4 Feb 2027, after the winners are announced, not on 29 Jan. we don't know the hour judging ends, and the app shouldn't change before the winners are out." The Stop_Time is `2027-02-04T08:00:00Z`, held in Config as text, because the Config_Loader refuses TOML dates. Nothing about the Website changes before then.
- **Q3 — The Website_Limits (Requirement 2, Requirement 10, Requirement 11, Requirement 17):** "10 website jobs and $50 a day, and $500 for the whole period." These are three limits in Config: 10 Website_Jobs a UTC_Day, $50 a UTC_Day, and $500 in total from the first deploy to the Stop_Time. They give the Rules_Steering's "daily job limit and a daily cost limit… the numbers come in the app spec" its numbers, and the total is a further limit on top. **Changed at review by R1:** the Daily_Cost_Limit is $52, so 10 fast Jobs still fit in a UTC_Day. The other two limits stay.
- **Q4 — The video check (Requirement 2, Requirement 8):** "phone videos usually put the moov box at the end of the file, so find it with range reads wherever it is, not only in the first few MB." The Video_Check is fresh pure-Python code, the Box_Reader, reading the MP4 and MOV structure over S3 range reads, with no ffprobe and no new package. It walks the top-level Boxes from the start of the file, skipping each one by its size, which covers 64-bit `largesize` Boxes and a size-0 Box that runs to the end of the file, until it finds `moov`, wherever that is. Then it reads the video's length and checks that a video track exists, matching what the Job_Maker's checks need. Its bounds, like the largest `moov` it reads and the most Boxes it walks, come from Config.

Round 2, the recommended answers the user took:

- **Q5 — How a Job counts against the limits (Requirement 10, Requirement 11):** a start reserves its Worst_Case_Cost, worked out from Config the way the Start_Command's quote does it: the dearest machine on the Preset's Fallback_List, run to its Time_Cap, plus its IPv4 address and disk. That's $4.94 for fast, $7.41 for default and $18.15 for quality, against expected costs of $1.23, $2.47 and $6.05. A Reservation is never settled down later, so the limits hold even when a Job hangs, and $50 a day covers 10 fast, 6 default or 2 quality Jobs. Days run in UTC. A start refused before any machine launches gives its Reservation back; a launch error that might have started a machine keeps it. Each Reservation is one atomic database write, so two judges pressing Start at once can't both get the last place. Only GPU machines count against the limits, not storage or downloads. **Changed at review by R1:** the Worst_Case_Cost also counts the 5 boot minutes in `[website].boot_margin_minutes` and rounds up to the next cent, so it's $5.15 for fast, $7.62 for default and $18.41 for quality, and $52 a day covers 10 fast, 6 default or 2 quality Jobs.
- **Q6 — Where the Website runs (Requirement 14, Requirement 16):** one AWS Lambda function on Python 3.11, which Lambda supports until 30 June 2027, with a function URL. That gives HTTPS on a `….lambda-url.us-east-1.on.aws` address with no domain, and the URL itself costs nothing. Judging costs about $0. No API Gateway, CloudFront, App Runner or EC2 server.
- **Q7 — Big files (Requirement 1, Requirement 2, Requirement 7, Requirement 12, Requirement 14):** Lambda takes at most 6 MB per request and returns at most 200 MB, so a 4 GB video can't pass through it. Only for a signed-in Account, the Website makes short-lived S3 links: the browser uploads the video straight to S3 in parts, each part's link valid for 15 minutes, and each download click gets a link to that one file, valid for 5 minutes. Both lifetimes go in Config. A link holds a signature, a temporary key id and a session token, never the secret key, and whoever has it can use it until it expires; the user accepted this under "no secret in the browser" and "a signed-out request never reaches a file". The Job_Bucket needs one CORS rule.
- **Q8 — Where accounts, sessions, the limit counters and the showcase live (Requirement 4, Requirement 12, Requirement 14):** one on-demand DynamoDB table, under $0.01 a month at this size. Sessions are random ids, stored hashed, that expire by themselves, so there's no signing secret and nothing goes in Secrets Manager or Parameter Store. The Read_Profile has no DynamoDB rights, so Kiro can't read the Password_Hashes. Job ownership lives in the table too, while a Job's status still comes from the Job's files in the Job_Bucket. **Changed at review by R3 and R4:** the table has point-in-time recovery and stays when the Stack is deleted or an update replaces it, and Ended without a result also needs a lookup of the Job's machines.
- **Q9 — Stack changes (Requirement 14, Requirement 15, Requirement 17):** the Website goes into `cloud/aws-setup.yaml`, which stays the one template (23.7 KB of its 51.2 KB limit), and the Account_Owner applies it with his usual change set in CloudShell. It adds the function, its URL and its log group, the table, a CORS rule and a lifecycle rule on the Job_Bucket, and a Website_Role. The Website_Role launches Job_Machines only from our tagged Machine_Image, under the Runner_Role's conditions. It reads the G_Quota, our machines, our images and the Job files, uploads only to `jobs/<Job_Id>/input/` and `uploads/`, and uses its own table. It can't terminate machines, make images, open shells or delete anything. The function's code is a template parameter, so a fresh copy's first create still works without it. **Changed at review by R2 and R3:** the function's code is two template parameters, the Website_Role may also call `cloudformation:DescribeStacks` on its own Stack, and the table is kept like the Job_Bucket.
- **Q10 — Storage and retention (Requirement 7, Requirement 10, Requirement 14, Requirement 17):** uploads land in `uploads/`, which S3 empties after 1 day. A started Job's video is copied to `jobs/<Job_Id>/input/` and kept like any Job's, until the Account_Owner clears the Website_Jobs after judging. The Website_Function's logs are kept 30 days. None of this counts against the Website_Limits.
- **Q11 — Who runs the new commands (Requirement 1, Requirement 3, Requirement 13, Requirement 14, Requirement 15, Requirement 17), changed by the user:** the Deploy_Command, the Account_Command and the Showcase_Command are the Account_Owner's alone. He runs them in CloudShell with the Admin_Login, like every stack change, because that login has no access keys. Kiro never runs them, so no new password reaches a chat, and both profiles keep exactly the rights they have now. The user's change, verbatim: "the deploy command has to fit in cloudshell, which keeps only 1 GB in the home folder, so it builds in /tmp and installs only what the deploy needs, not the whole uv sync." So the Deploy_Command keeps its clone, its build, its caches and the Website_Bundle under `/tmp`, and never runs a `uv sync` of the root project (numpy, scipy, OpenCV and the dev tools). CloudShell commands name only the region, never a profile, as the Setup_Guide already says. Two points follow from the user's answer and are flagged here rather than asked: the Account_Command and the Showcase_Command run in the same CloudShell, so they too need only the Python standard library and boto3, installed the same way; and because an AWS client with no profile on a laptop would use whatever keys the laptop's environment holds, each CloudShell_Command refuses to run outside CloudShell.
- **Q12 — The one yes (Requirement 10, Requirement 14, Requirement 15, Requirement 16, Requirement 17):** the Deploy_Command calls the Spend_Guard with the Website_Limits' worst case, "website GPU jobs: up to 10 and $50 a day, $500 in total, until 2027-02-04 08:00 UTC", so `APPROVED_SPEND=1` on the Deploy_Command is the Account_Owner's one yes. Changing a limit means a redeploy and a new yes, and a redeploy doesn't reset the counters. The Website never holds `APPROVED_SPEND` in its environment: it calls the same `check_spend` only after the limit check has taken the Job's Reservation. The Rules_Steering and the AWS_Jobs_Skill get updated to say so. **Changed at review by R1 and R7:** the worst case is split across `check_spend`'s three fields and says $52: "website GPU jobs (up to 10 and $52 a UTC day)", "every day until 2027-02-04 08:00 UTC" and "at most $500 in total".
- **Q13 — The Showcase (Requirement 13):** the Account_Owner adds and removes Showcase_Entries with the Showcase_Command, and an entry is live at once, with no redeploy. Any finished Job can go in, from the Website or from a laptop. Each entry has a title. An entry made from CC BY or CC BY-SA footage carries the author's credit and the licence, shown beside it, as the Rules_Steering requires.
- **Q14 — How the Website uses "the same code as cloud.start" (Requirement 9, Requirement 15):** `start_job` splits into one shared Start_Core and two thin callers. The Start_Core holds the Step_List check, the newest Machine_Image, the lock-hash check, the Smoke_Marker, the machine choice and the G_Quota, the Spend_Guard, the uploads and the fallback launch. The Laptop_Caller keeps today's git checks and `make_job`. The Website_Caller runs the Baked_Commit: the Deploy_Command refuses a dirty tree, as the Start_Command does, and makes the Code_Archive and the lock-file hashes from that commit, so after a new Machine_Image with a different `uv.lock`, Website starts refuse until a redeploy. The Website builds the Job_Record from the Pipeline's public pieces (the Job_Id, the Preset and Trim checks, and `write_record`), with no change to `splatpipe/`, and copies the video inside S3, not to disk. The cloud-jobs tests keep passing unchanged.
- **Q15 — No web framework (Requirement 1, Requirement 17, Requirement 18):** the handler, the routing, the pages, the forms, the cookies and the sign-in use only the Python standard library, and AWS goes through `cloud/`, as the Tech_Steering says (boto3 only inside `cloud/`). So the App_Project holds only boto3 for the Website_Bundle, and the Licence_List needs no new row. The tests live in `tests/` and run in the root environment, which already holds boto3, pytest and Hypothesis. **Changed at review:** the App_Project pins boto3 and the packages boto3 pulls in to the root `uv.lock`'s versions, and the Licence_List gains rows for Python, which runs the CloudShell_Commands and the Website, and pip, which installs the Website_Bundle's packages in CloudShell.

### Assumptions to confirm at review

The user didn't ask for these in the request. A to I are the recommended answers, which the clarify step accepted for the review to confirm. J to M came up while writing these requirements and haven't been put to the user yet. Each is marked in the requirement it belongs to, and each is a small change if the review decides otherwise. The review confirmed A to M (R11), except that R4 replaces C's Time_Cap rule.

- **A — Pages (Requirement 6, Requirement 7):** the pages are rendered on the server, with a little plain JavaScript for the part upload and the status refresh, and no Node build. They work on a phone or a laptop. The start form offers fast (chosen when the page opens), default and quality, each with its Expected_Time.
- **B — No trim on the Website (Requirement 8, Requirement 9):** a Website_Job uses the whole video. The Start_Command keeps `--start` and `--end`.
- **C — The status words (Requirement 12):** Starting, Running (step n of N), Done, Done with warnings, and Failed (with the Job's Readable_Error). Once the Time_Cap has passed with no Job_Summary, the status is Ended without a result. A running Job's status refreshes every `[cloud].poll_seconds`. **Replaced at review by R4:** a Job with no Job_Summary is Ended without a result only when at least 10 minutes have passed since it started, its machines are terminated or can't be found, and a second read still finds no Job_Summary.
- **D — Busy and can't start (Requirement 11):** "every GPU type is full" means `admit`'s G_Quota refusal or the launch loop's no-capacity error. Both show "All GPU machines are busy right now, so your job didn't start. Try again later.", both give the Reservation back, and nothing is queued. For problems a judge can't fix (no Machine_Image, no Smoke_Marker, a lock-hash drift, a G_Quota of 0), the Website shows "The website can't start jobs right now." and logs the reason, redacted.
- **E — Sign-in limits and cookies (Requirement 2, Requirement 4):** after 5 failed sign-ins from one address in 15 minutes, that address waits 15 minutes. The limit is per address, not per Account, so nobody can lock the judges' shared Account. Sessions last 7 days. Cookies are `__Host-`, Secure, HttpOnly and SameSite=Lax. Every form post carries a CSRF_Token, and there's a sign-out button. All these numbers go in Config.
- **F — Signed-out requests (Requirement 5):** signed out, only the sign-in page, its form post, its stylesheet and the favicon answer. Everything else, the Showcase included, redirects to the sign-in page, or returns 401 to the Page_Script. None of these requests makes an AWS call or a link (but see L).
- **G — Who sees which Jobs (Requirement 12):** an Account sees only the Jobs it started on the Website. Jobs started from a laptop appear only through the Showcase.
- **H — The Result_Page (Requirement 12, Requirement 17):** the Result_Page shows the status, the Preset, the timings and one download button, for `out/clean.sog`, or `out/train.ply` when there's no clean copy. It also has one empty, labelled Viewer_Slot for Aarsh's viewer. The viewer's splat-transform pin, which the Clean_Spec left to "the app spec", moves to the Viewer_Spec.
- **I — After the Stop_Time (Requirement 11, Requirement 16, Requirement 17):** everything except starting Jobs keeps working until the Account_Owner takes the Website down. The Product_Steering gets a paragraph about the Website.
- **J — Replacing and removing Accounts (Requirement 3), new:** the Account_Command can also give an existing Account a new password, ending its Sessions, and remove an Account. The request only says the Account_Owner makes Accounts with a command. Without J, a leaked judges' password could only be replaced by making an Account under a new name. Dropping J removes Requirement 3's criteria 6 to 8.
- **K — The Start_Precheck (Requirement 7), new:** when an Account presses Start, the Website runs the Start_Precheck before the first part of the Upload goes, reserving nothing and starting nothing, so a judge on a phone doesn't send up to 4 GB for a Job that can't start. The start itself runs every check again.
- **L — The one table read of a signed-out request (Requirement 5), new:** the request asks that a signed-out request make no AWS call. A Session is a random id that only the Website_Table holds, so finding out that a well-formed Session_Cookie names no Valid_Session takes one read of the Website_Table. That read is the only AWS call a signed-out request outside the Open_Routes makes. A request with no Session_Cookie, or a malformed one, makes no AWS call at all.
- **M — A launch refused for the G_Quota (Requirement 10, Requirement 11), new:** when AWS refuses a launch because the G_Quota is full (another start took the room after `admit` let this one through), no machine started. So the Website treats it like the no-capacity error of D: it shows the busy message and gives the Reservation back. Q5 gives a Reservation back for any start refused before a machine launches, and D names only `admit`'s refusal and the no-capacity error.

### Settled at review

The team's reviewer agent reviewed the spec committed on `feat/website`, and the user answered the review. R1 to R6 are the user's own answers, quoted. R7 to R11 are the recommendations the user took with "the rest as recommended". D2 came from fixing one of the reviewer's defects and added behaviour the user hadn't been asked about, so it was flagged for the user, who confirmed it in a second answer, quoted. R12 is a rule the user added in that second answer, quoted. The reviewer's other defects are fixed in the criteria they touch. Each item is marked in the requirement it changes.

- **R1 — The boot minutes (Requirement 2, Requirement 10, Requirement 15, Requirement 17):** "reserve the 5 minutes, and raise the daily limit to $52 so 10 fast jobs a day still fit. the total stays $500, and the deploy's spend-guard message says $52." The Worst_Case_Cost becomes `rough_cost` of the dearest type on the Preset's whole Fallback_List, run for the Time_Cap plus `[website].boot_margin_minutes` (5), plus the IPv4 address and the root volume for that time, rounded up to the next cent after first rounding away float noise. The boot minutes count because a hung machine runs them before the Time_Cap's shutdown is scheduled. With today's Config that's 515 cents ($5.15) for fast, 762 ($7.62) for default and 1841 ($18.41) for quality. The Daily_Cost_Limit becomes $52, which covers 10 fast Jobs (5150 cents), 6 default (4572) or 2 quality (3682) in a UTC_Day, and $500 covers 97 fast, 65 default or 27 quality. The Deploy_Command's Spend_Guard strings become "website GPU jobs (up to 10 and $52 a UTC day)", "every day until 2027-02-04 08:00 UTC" and "at most $500 in total". The laptop Start_Command's quote doesn't change. This changes Q3 and Q5.
- **R2 — Two code parameters and DescribeStacks (Requirement 14, Requirement 15, Requirement 17):** "yes to both. the code bucket parameter names our own job bucket, with the website's code under its own prefix, never another bucket in the account. the stack is still deployed with --template-body, so the template stays under 51,200 bytes." The Website_Code_Parameters are `WebsiteCodeBucket` and `WebsiteCodeKey`. The Deploy_Command fills `WebsiteCodeBucket` only with the Stack's own `BucketName` output and `WebsiteCodeKey` only as `website/<commit id>.zip`, and refuses anything else. The Stack_Template lets `WebsiteCodeKey` be only empty or `website/<40 hex characters>.zip`. The Setup_Guide's commands never type a bucket name. The Deploy_Command sends the Stack_Template inline, as the change set's template body and never a template URL, and refuses a Stack_Template over 51,200 bytes before it uploads anything. The Website_Role may call `cloudformation:DescribeStacks` on its own Stack only, which the Website uses to find its resources at run time. This changes Q9.
- **R3 — Point-in-time recovery and the kept table (Requirement 14, Requirement 17):** "yes to both, with the same DeletionPolicy and UpdateReplacePolicy as the bucket: RetainExceptOnCreate and Retain." The Website_Table has point-in-time recovery on, `DeletionPolicy: RetainExceptOnCreate` and `UpdateReplacePolicy: Retain`, and the Setup_Guide's "Removing the setup" keeps it. This changes Q8 and Q9.
- **R4 — A machine that dies without a Job_Summary (Requirement 2, Requirement 12, Requirement 18):** "yes, but show 'Ended without a result' only once the machine is terminated or can't be found, at least 10 minutes after the job started, and after checking once more for the summary, so a machine that's still launching or has just finished isn't counted as dead." A Job with no Job_Summary is Ended without a result only when all of these hold: at least `[website].ended_check_minutes` (10) have passed since the Job started; the Website finds every machine carrying the Job's Job_Id tag and the Project_Tag terminated, or finds none; and a second read of `out/job.json` after that lookup still finds no Job_Summary. If the lookup fails, the Job keeps its Starting or Running status. The lookup is a describe of EC2, which Requirement 14's criterion 9 already allows. This replaces assumption C's Time_Cap rule, and changes Q8.
- **R5 — No AWS from this laptop (a task rule):** "this laptop has no aws access for this project. if a task ever needs a real aws call, stop and ask." The tasks get this rule later. It changes no requirement.
- **R6 — Pushing each commit (a task rule):** "keep pushing the changes done so my commits will be visible." Each task pushes `feat/website` after its commit. The tasks get this rule later. It changes no requirement.
- **R7 — The design's other points for review stand (Requirement 3, Requirement 5, Requirement 9, Requirement 10, Requirement 11, Requirement 15):** the Session_Cookie carries the Account's name. One Upload starts at most one Job. Removing an Account deletes its items in the Website_Table, and its files in the Job_Bucket stay. `QuotaRefused` covers EC2's `InstanceLimitExceeded` as well as `VcpuLimitExceeded`. The Spend_Guard's text is split across `check_spend`'s three fields, with R1's $52. This changes Q12.
- **R8 — The team using the Website (Requirement 16, Requirement 17):** Jobs started through the Website count against the Website_Limits whoever starts them, the team included, and the Account_Owner's one yes covers them. Jobs started by hand with `cloud.start` still need their own yes. The Rules_Steering and the AWS_Jobs_Skill say so.
- **R9 — A hook block (Requirement 1, Requirement 2, Requirement 17, Requirement 18):** the Guard_Hook also blocks any shell command that runs one of the CloudShell_Commands' modules on a laptop. The modules are named in `[website].cloudshell_modules`, and `tests/test_guard.py` gets the cases. This is a second layer on top of the CloudShell check, which reads `AWS_EXECUTION_ENV`: AWS doesn't document that variable, and anyone can set it. The Setup_Guide tells the Account_Owner to check that `echo $AWS_EXECUTION_ENV` prints `CloudShell` before running a CloudShell_Command.
- **R10 — The judges' videos (Requirement 17):** no AWS right changes. The Setup_Guide's website section says that the Project_Group's Read_Policy, and so the Read_Profile that Kiro uses, can read `uploads/` and the Website_Jobs' inputs. R12 adds that Kiro never opens or downloads them.
- **R11 — The assumptions:** A to M are confirmed, except that R4 replaces C's Time_Cap rule.
- **R12 — Kiro never opens the videos (Requirement 17):** "one thing is missing from 5: also add a line to rules.md, and the same line to the aws-jobs skill, that kiro never opens or downloads anything under uploads/ or any job's input/ in the bucket, because those are judges' and users' videos. kiro never needs them: status and downloads only read out/." The line goes in the Rules_Steering and the AWS_Jobs_Skill, in the same words in both. The Status_Command and the Download_Command read only a Job's `out/` and its reports, so Kiro never needs `uploads/` or any `input/`. No AWS right changes, as R10 says.
- **D2 — The second clock read (Requirement 10):** "yes to the stop-time check right before the launch." Right before the launch, after the copy inside S3 and the uploads, the Website reads the clock again. If the Stop_Time has passed, the Website gives the Reservation back, launches nothing and shows the Stop_Time message. It comes from the reviewer's defect about the clock, whose other fix needed no confirming: the Reservation's Stop_Time check and its UTC_Day come from a clock read right before the Reservation's transaction, not from when the request arrived.

## Glossary

Terms this spec reuses from the pipeline-core spec and the Pipeline_Doc: Pipeline, Step, Step_List, Required_Step, Optional_Step, Job, Job_Id, Job_Record, Job_Maker (`make_job`), Video_Length, Trim, Used_Part, Step_Report, Job_Runner (`run_job`), Job_Summary, Job_Log, Job_Status, Preset, Config, Config_Loader, Pipeline_Section, Readable_Error, Spend_Guard, Spend_Variable, Paid_Entry_Point, Pipeline_Doc, Licence_List and Test_Suite.

Terms this spec reuses from the aws-setup spec: Stack_Template, Stack, Stack_Region, Stack_Output, Account_Owner, Admin_Login, CloudShell, Job_Bucket, Bucket_Job_Folder, Bucket_Work_Folder, Job_Role, Job_Profile, Job_Machine, Job_Security_Group, Default_VPC, DefaultVpcId, Runner_Role, Read_Policy, Read_Profile, Runner_Profile, Project_Group, Project_Tag, Machine_Image, Allowed_Type, G_Quota, IMDSv2, Other_Resource, Compliance_Check, Unfit_Rule, Suppression, Template_Test, Builder, Setup_Guide, Rules_Steering, Tech_Steering, Structure_Steering, README and Project_File.

Terms this spec reuses from the cloud-jobs spec: Cloud_Jobs, Start_Command, Status_Command, Download_Command, Smoke_Test, Smoke_Command, Smoke_Marker, Image_Build_Command, Cloud_Runner, Code_Archive, Time_Cap, Expected_Time, Step_Needs, Machine_Spec, Fallback_List, Account_Redaction, Cloud_Config, Fake_AWS and AWS_Jobs_Skill.

New terms this spec defines:

- **Website**: the web app this spec builds, in `app/`, that judges and other users open in a browser on a laptop or a phone to sign in, upload a video, start a Job, follow it and download its result. It runs as the Website_Function.
- **Website_Function**: the one AWS Lambda function, on Lambda's Python 3.11 runtime, that runs the Website.
- **Website_URL**: the Website_Function's function URL, an HTTPS address of the form `https://<id>.lambda-url.<Stack_Region>.on.aws/` that Lambda gives the function, so the Website has HTTPS with no domain to buy.
- **Website_Role**: the IAM role that the Website_Function runs as.
- **Website_Table**: the one on-demand DynamoDB table that the Stack makes for the Website. It holds the Accounts, the Sessions, the Sign_In_Failures, the Uploads, the Limit_Counters, the Job_Owners and the Showcase_Entries.
- **Website_Bundle**: the zip file the Deploy_Command builds, holding everything the Website_Function runs.
- **Website_Code_Parameters**: the Stack_Template's two parameters that name where the Website_Bundle is: `WebsiteCodeBucket`, which names our own Job_Bucket, and `WebsiteCodeKey`, the Website_Bundle's key under `website/` in it. Both are empty until the first deploy.
- **Placeholder_Function**: the small inline code the Stack_Template gives the Website_Function while the Website_Code_Parameters are empty. It answers every request with a page saying the website isn't deployed yet.
- **App_Project**: `app/pyproject.toml`, with `app/uv.lock` next to it: the Website's own Python project, whose only dependency is boto3, held with the packages boto3 pulls in at the root `uv.lock`'s versions.
- **Account**: a name and a Password_Hash in the Website_Table. The Account_Owner makes each Account with the Account_Command, and the judges share one.
- **Account_Name_Pattern**: the pattern an Account's name must match, 1 to 32 characters of lowercase ASCII letters, digits and hyphens, starting with a letter. It stays in code because it's a safety check, like the Job_Id pattern.
- **Password_Hash**: what `hashlib.scrypt` makes from an Account's password with a random salt and the scrypt settings in `[website]`, stored with that salt and those settings.
- **Session**: one sign-in of an Account. Its id is a random value that the browser holds in the Session_Cookie and that the Website_Table holds only as its SHA-256 hash, with the Account, the Session's expiry and the Session's CSRF_Token.
- **Session_Cookie**: the cookie that holds the Account's name and the Session's id, named with the `__Host-` prefix and set Secure, HttpOnly and SameSite=Lax, with Path `/` and no Domain.
- **Valid_Session**: a Session that the Website_Table holds, that hasn't been ended, whose expiry hasn't passed and whose Account still exists.
- **CSRF_Token**: a random value that every form post, and every Script_Request that changes something, must carry: the Session's own token, or on the sign-in page, a token that the page sets for its own form.
- **Sign_In_Address**: the address a request comes from, as the Website_URL reports it in the request's context, never as a header of the request says.
- **Sign_In_Failure**: one failed sign-in from one Sign_In_Address, counted in the Website_Table.
- **Open_Route**: one of the four requests the Website answers without a Valid_Session: GET of the sign-in page, POST of the sign-in form, GET of the stylesheet and GET of the favicon.
- **Page_Script**: the plain JavaScript file that the Website's signed-in pages load, with no build step, for the part upload and the status refresh.
- **Script_Request**: a request that the Page_Script sends, as opposed to a page that the browser opens.
- **Presigned_Link**: a short-lived S3 link that the Website makes with the Website_Role's temporary credentials. It holds a signature, a temporary key id and a session token, never the secret key, and whoever holds it can use it until it expires.
- **Upload**: one video that the Page_Script sends straight to `uploads/` in the Job_Bucket as an S3 multipart upload, recorded in the Website_Table with the Account that began it.
- **Upload_Link**: a Presigned_Link for sending one part of an Upload, valid for `[website].upload_link_seconds`.
- **Download_Link**: a Presigned_Link for fetching one Result_File, valid for `[website].download_link_seconds`.
- **Box**: one unit of the MP4 and MOV file structure: a 32-bit big-endian size and a 4-character type, then a 64-bit `largesize` when the size is 1. A size of 0 means the Box runs to the end of the file.
- **Box_Reader**: the Website's pure-Python reader of an Upload's Boxes, over S3 range reads, with no ffprobe and no new package.
- **Video_Check**: the Website's checks of an Upload, in place of the Job_Maker's: the extension, the size, the Box_Reader's video track and Video_Length, and the Used_Part.
- **Start_Core**: the shared part of starting a Job that this spec splits out of the Start_Command's `start_job`: the Step_List check, the newest available Machine_Image, the lock-hash check, the Smoke_Marker check, the machine choice against the G_Quota, the money quote, the Spend_Guard, the uploads and the fallback launch.
- **Laptop_Caller**: the Start_Command's `start_job` after the split. It keeps the uncommitted-change check, the Code_Archive and the lock hashes of `HEAD` and `make_job`, and calls the Start_Core.
- **Website_Caller**: the Website's start of a Job. It runs the Baked_Commit, the Video_Check, the Job_Record, the Reservation and the copy inside S3, and calls the Start_Core.
- **Baked_Commit**: the commit the Deploy_Command built the Website_Bundle from, carried in the Website_Bundle as the commit's id, its Code_Archive and the hashes of its `uv.lock` and `package-lock.json`.
- **Start_Precheck**: the free checks that need no video, which the Website runs when an Account presses Start, before the Upload's first part goes (assumption K): the Stop_Time, the room in the Website_Limits for the chosen Preset's Reservation, and the Start_Core's checks up to the money quote. It takes no Reservation and starts nothing.
- **Website_Job**: a Job that an Account started on the Website.
- **Job_Owner**: the Account that started a Website_Job, as the Website_Table records it.
- **Website_Limits**: the three limits on the GPU machines the Website starts, set in `[website]`: the Daily_Job_Limit, the Daily_Cost_Limit and the Total_Cost_Limit.
- **UTC_Day**: one calendar day in UTC, from 00:00 to the next 00:00.
- **Daily_Job_Limit**: at most 10 Reservations in one UTC_Day.
- **Daily_Cost_Limit**: at most $52 of Reservations in one UTC_Day.
- **Total_Cost_Limit**: at most $500 of Reservations in total, from the first deploy to the Stop_Time, across every redeploy.
- **Worst_Case_Cost**: the most one Job of a Preset can cost: the dearest Allowed_Type on the Preset's whole Fallback_List, run for the Preset's Time_Cap plus `[website].boot_margin_minutes` (5 minutes, because a hung machine runs its boot minutes before the Time_Cap's shutdown is scheduled), plus the public IPv4 address and the root volume for that time, worked out with the Start_Command's `rough_cost` and rounded up to the next cent after first rounding away float noise. With today's Config it's $5.15 for fast, $7.62 for default and $18.41 for quality.
- **Reservation**: one Website_Job's place in the Daily_Job_Limit and its Worst_Case_Cost in the Daily_Cost_Limit and the Total_Cost_Limit, taken in one atomic write when the Job starts and never settled down later.
- **Limit_Counters**: the counts in the Website_Table that the Website_Limits are checked against: each UTC_Day's number of Reservations and reserved cents, and the total reserved cents.
- **Stop_Time**: the instant in `[website].stop_starting_at`, `2027-02-04T08:00:00Z`, from which the Website starts no more Jobs.
- **Website_Status**: the status the Website shows for a Job: Starting, Running (step n of N), Done, Done with warnings, Failed, or Ended without a result.
- **Result_Page**: the Website's page for one Job.
- **Result_File**: the one file of a Job that the Result_Page offers to download: `out/clean.sog` (the name in `[clean].clean_sog`), or `out/train.ply` (the name in `[clean].train_ply`) when the Job's `out/` holds no `clean.sog`.
- **Viewer_Slot**: the one empty, labelled place on the Result_Page where the Viewer_Spec will put the 3D viewer.
- **Viewer_Spec**: Aarsh's later spec for the 3D viewer.
- **Showcase**: the Website's page that lists the Showcase_Entries.
- **Showcase_Entry**: a finished Job that the Account_Owner put in the Showcase, with a title and its Footage_Licence, and for CC BY or CC BY-SA footage the author's credit.
- **Footage_Licence**: the licence of the video a Showcase_Entry was made from: our own video, CC0, CC BY, CC BY-SA or the Pexels licence, the list the Rules_Steering allows.
- **Account_Command**: the command that makes an Account, and replaces an Account's password or removes an Account (assumption J). One of the CloudShell_Commands.
- **Showcase_Command**: the command that adds, lists and removes Showcase_Entries. One of the CloudShell_Commands.
- **Deploy_Command**: the command that builds the Website_Bundle under `/tmp` in CloudShell, uploads it and makes the Stack's change set. A Paid_Entry_Point, and one of the CloudShell_Commands.
- **CloudShell_Commands**: the Deploy_Command, the Account_Command and the Showcase_Command, which only the Account_Owner runs, in CloudShell with the Admin_Login.
- **CloudShell_Steps**: the Setup_Guide's steps that put a clone of the repo, Python 3.11 and boto3 under `/tmp` in CloudShell and run a CloudShell_Command from that clone.
- **Product_Steering**: `.kiro/steering/product.md`.
- **Cloud_Steering**: `.kiro/steering/cloud.md`.
- **Guard_Hook**: the Kiro hook `scripts/hooks/guard.py`, which `.kiro/hooks/guard.json` runs before every shell command Kiro sends, and which blocks the commands our rules forbid.
- **Clean_Spec**: the clean step's spec, `.kiro/specs/clean/requirements.md`.

## Requirements

### Requirement 1: Where the Website lives and how it reaches AWS

**User Story:** As a team member, I want the Website's code in `app/` with its own small Python project, reaching AWS only through `cloud/`, so that a Website package never changes the root `uv.lock` or forces a new Machine_Image, and the Pipeline never learns about the Website.

**Settled in clarification:** Q7 (criterion 10), Q11 (criteria 8 and 9), Q15 (criteria 3 and 4). **Settled at review:** R9 (criterion 11).

#### Acceptance Criteria

1. THE Website's code SHALL live under `app/`, and THE Website's tests SHALL live in `tests/` and run in the root environment with `uv run pytest`, which already holds boto3, pytest and Hypothesis.
2. THE App_Project SHALL list boto3 as its only dependency, pinned to the root `uv.lock`'s version, and SHALL hold each package boto3 pulls in at the root `uv.lock`'s version, so `app/uv.lock` holds only boto3 1.44.9, botocore 1.44.9, s3transfer 0.19.3, jmespath 1.1.0, python-dateutil 2.9.0.post0, six 1.17.0 and urllib3 2.8.0 with today's root `uv.lock`, and the Website_Bundle runs the same boto3 as the Test_Suite; and THE Project_File and the root `uv.lock` SHALL stay exactly as they are, because a Job refuses to start when the root `uv.lock` differs from the Machine_Image's, so a new root package would mean a new Machine_Image.
3. THE Website SHALL build its request handler, routing, pages, forms, cookies and sign-in from the Python standard library alone, with no web framework (Q15).
4. THE Website SHALL reach AWS only through the `Aws` protocol in `cloud/aws.py`, and THE files under `app/` SHALL import neither boto3 nor botocore, so boto3 stays inside `cloud/` as the Tech_Steering says (Q15).
5. THE Website SHALL call only the public functions and names of `cloud/` and `splatpipe/` and SHALL change no file under `splatpipe/`, and THE files under `splatpipe/` SHALL import neither `app` nor `cloud`.
6. WHEN the Website runs on the Website_Function, THE Website SHALL get its `Aws` from a builder in `cloud/aws.py` that names the region from `[aws].region` and no profile, so the Website_Function uses the Website_Role.
7. THE Website SHALL look up only our own AWS resources, by the Project_Tag or by a name or id the Stack gives, as Cloud_Jobs does, and SHALL find the Job_Bucket, the Website_Table, the Job_Profile, the Job_Security_Group and the Default_VPC from the Stack, never from a name, id or ARN kept in the repo.
8. WHEN a CloudShell_Command makes an AWS client, THE CloudShell_Command SHALL name the region from `[aws].region` and no profile, because CloudShell runs as the Admin_Login and has no named profiles, as the Setup_Guide says for every CloudShell command (Q11).
9. IF a CloudShell_Command starts anywhere other than CloudShell, which sets the environment variable `AWS_EXECUTION_ENV` to `CloudShell`, THEN THE CloudShell_Command SHALL stop with a Readable_Error that says it runs only in CloudShell with the Admin_Login, before it makes any AWS client, because a client with no profile on a laptop would use whatever AWS keys the laptop's environment holds (Q11, Rules_Steering).
10. THE code and the pages that the Website sends to a browser SHALL hold no AWS key, no password, no Password_Hash and no other secret, and the only AWS credential parts that reach a browser SHALL be the signature, the temporary key id and the session token inside a Presigned_Link (Q7).
11. WHEN Kiro sends, on a laptop, a shell command that runs one of the modules named in `[website].cloudshell_modules`, THE Guard_Hook SHALL block the command, as a second layer on top of criterion 9's check, because AWS doesn't document `AWS_EXECUTION_ENV` and anyone can set it (R9).

### Requirement 2: The `[website]` Config section

**User Story:** As the Account_Owner, I want every number and setting the Website uses in Config, so that the limits, the Stop_Time, the link lifetimes and the sign-in rules change without a code change.

**Settled in clarification:** Q2 (criterion 2), Q3 (criterion 3), Q4 (criterion 6), Q7 (criterion 4). **Assumptions to confirm at review:** E (criterion 5). **Settled at review:** R1 (criteria 3, 6 and 11), R4 (criterion 6), R9 (criterion 6).

#### Acceptance Criteria

1. THE Config SHALL hold a `[website]` Pipeline_Section, so the Config_Loader refuses NaN, infinity and TOML dates or times in it, and THE Config SHALL give every key in `[website]` a comment on the line above the key, which the Test_Suite's existing comment check covers.
2. THE `[website]` section SHALL hold the Stop_Time as the text `"2027-02-04T08:00:00Z"`, because the Config_Loader refuses TOML dates in a Pipeline_Section (Q2).
3. THE `[website]` section SHALL hold the Website_Limits: 10 Reservations a UTC_Day, $52 a UTC_Day and $500 in total (Q3, R1).
4. THE `[website]` section SHALL hold the Upload_Link lifetime (900 seconds), the Download_Link lifetime (300 seconds) and the size of each Upload part (Q7).
5. THE `[website]` section SHALL hold the sign-in numbers of assumption E (5 Sign_In_Failures within 15 minutes, a 15-minute wait and a 7-day Session), the scrypt settings and the length of the passwords the Account_Command makes.
6. THE `[website]` section SHALL hold the Box_Reader's bounds, the most Boxes the Box_Reader walks and the largest `moov` Box it reads (Q4), the Worst_Case_Cost's boot margin `boot_margin_minutes` (5) (R1), the wait `ended_check_minutes` (10) before a Job can be Ended without a result (R4), the modules of the three CloudShell_Commands in `cloudshell_modules`, which the Guard_Hook reads (R9), and every other number and setting the Website reads, like the most Jobs the job list shows, the longest Showcase title, the longest file name kept, the longest request body and the folder under `/tmp` that the Deploy_Command builds in, so the Website's code holds no hard-coded number.
7. THE Website SHALL read each setting that another section already holds from that section and SHALL copy none into `[website]`: the poll interval from `[cloud].poll_seconds`, the default Preset and the video limits from `[jobs]`, the Result_File names from `[clean]`, the Fallback_Lists, Expected_Times and Time_Caps from `[cloud.presets]`, and the prices from `[cloud.types]` and `[cloud]`.
8. IF a `[website]` value is missing or has the wrong type, or the Stop_Time isn't a UTC instant written as `YYYY-MM-DDTHH:MM:SSZ`, THEN THE Website's Config reader SHALL refuse with a Readable_Error that names the key and says what it must be.
9. THE Stack_Template SHALL hold the numbers that only the Stack uses (the 1-day `uploads/` expiry, the 30-day log retention, and the Website_Function's memory and timeout), as the Stack_Template's header says for its own numbers, because CloudFormation can't read Config.
10. THE Config's `[git]`, `[aws]`, `[paid]` and `[devlog]` sections SHALL stay as they are, so `tests/test_config.py`'s check of those sections still passes.
11. THE Test_Suite SHALL check that `[website]` loads from the real Config and holds the settled values: the Stop_Time, the three Website_Limits with the $52 Daily_Cost_Limit, the 5-minute boot margin, the two link lifetimes and assumption E's numbers (R1).

### Requirement 3: Accounts and the Account_Command

**User Story:** As the Account_Owner, I want to make the Website's Accounts with a command that makes long random passwords only I see, so that there's no sign-up page and no password anyone chose can be guessed.

**Settled in clarification:** Q1 (criteria 2, 3 and 4), Q11 (criterion 9). **Assumptions to confirm at review:** J (criteria 6, 7 and 8). **Settled at review:** R7 (criterion 8).

#### Acceptance Criteria

1. THE Website SHALL have no page and no request that makes an Account, so the Account_Command is the only way to make one, and THE Setup_Guide SHALL tell the Account_Owner to make one Account that every judge shares.
2. WHEN the Account_Owner runs the Account_Command with a new name that matches the Account_Name_Pattern, THE Account_Command SHALL make the Account with a new password drawn from Python's `secrets` module, of the length in `[website]` (Q1).
3. THE Account_Command SHALL store only the Password_Hash, made with `hashlib.scrypt`, a new random salt and the scrypt settings in `[website]`, with the salt and the settings stored beside the Password_Hash, so a later change of the settings still checks the Accounts made before it (Q1).
4. THE Account_Command SHALL print the new password once, on the terminal, and SHALL write the password to no file, no log and no AWS service (Q1).
5. IF the name doesn't match the Account_Name_Pattern, THEN THE Account_Command SHALL refuse with a Readable_Error that gives the pattern.
6. IF the Account_Owner makes an Account whose name is taken, THEN THE Account_Command SHALL refuse with a Readable_Error that names the option for replacing the password, and SHALL change nothing (assumption J).
7. WHEN the Account_Owner asks the Account_Command to replace an Account's password, THE Account_Command SHALL give the Account a new password, print the new password once and end every Session of that Account (assumption J).
8. WHEN the Account_Owner asks the Account_Command to remove an Account, THE Account_Command SHALL remove the Account and the Account's own items in the Website_Table, its Sessions, its Uploads and the Job_Owner records of its Website_Jobs, so every Session of that Account ends, and SHALL leave that Account's files in the Job_Bucket (assumption J, R7).
9. THE Account_Command SHALL need only the Python standard library and boto3, and SHALL run only in CloudShell, as Requirement 1 says (Q11).

### Requirement 4: Signing in, Sessions and signing out

**User Story:** As a judge, I want to sign in with the shared Account and stay signed in for a week, so that I can come back to my Jobs without signing in each time, while nobody can guess their way in or lock the others out.

**Settled in clarification:** Q1 (criterion 1), Q8 (criteria 7, 9 and 11). **Assumptions to confirm at review:** E (criteria 4, 5, 7, 8 and 12).

#### Acceptance Criteria

1. WHEN someone posts the sign-in form with an Account's name and password, THE Website SHALL hash the password with `hashlib.scrypt`, the Account's salt and the Account's stored scrypt settings, compare the result with the Password_Hash in constant time, and make a Session when the two match (Q1).
2. WHEN someone posts the sign-in form with a name that has no Account, THE Website SHALL still run one scrypt hash with the scrypt settings in `[website]` and give the same refusal as for a wrong password, so neither the answer nor its timing shows which names have an Account.
3. IF the name or the password is wrong, THEN THE Website SHALL refuse the sign-in with a Readable_Error that says the name and the password don't match an Account, count one Sign_In_Failure for the Sign_In_Address, and make no Session.
4. IF a Sign_In_Address has 5 Sign_In_Failures within 15 minutes, THEN THE Website SHALL refuse every sign-in from that Sign_In_Address for the next 15 minutes without checking any password, with a Readable_Error that says how many minutes to wait, all three numbers from `[website]` (assumption E).
5. THE Website SHALL count Sign_In_Failures per Sign_In_Address and never per Account, so nobody can lock the judges' shared Account by failing on purpose (assumption E).
6. THE Website SHALL take the Sign_In_Address only from the request's source address as the Website_URL reports it, and never from a header the browser sends, like `X-Forwarded-For`.
7. WHEN the Website makes a Session, THE Website SHALL draw the Session's id and its CSRF_Token from Python's `secrets` module, store the id only as its SHA-256 hash in the Website_Table with the Account, the CSRF_Token and an expiry 7 days later, and send the Session_Cookie (Q8, assumption E).
8. THE Session_Cookie SHALL have a name with the `__Host-` prefix and the attributes Secure, HttpOnly, SameSite=Lax and Path=/, with no Domain (assumption E).
9. THE Website_Table SHALL remove an expired Session by itself, through a time-to-live attribute, and THE Website SHALL treat a Session whose expiry has passed as signed out whether or not the Website_Table has removed it yet, because DynamoDB removes an expired item some time after it expires (Q8).
10. WHEN a signed-in Account presses the sign-out button, THE Website SHALL end the Session in the Website_Table at once, clear the Session_Cookie and show the sign-in page.
11. THE Website SHALL keep no signing secret and SHALL put nothing in Secrets Manager or Parameter Store (Q8).
12. IF a form post, or a Script_Request that changes something, carries no CSRF_Token or a CSRF_Token that doesn't match the request's Session (or, for the sign-in form, the sign-in page's own token), THEN THE Website SHALL answer HTTP 403 and change nothing (assumption E).
13. THE Website SHALL change an Account, a Session, an Upload or a Job only on a POST request, so opening a page or following a link changes nothing.

### Requirement 5: Signed-out requests

**User Story:** As the Account_Owner, I want every page except sign-in to need a signed-in Account, so that nobody without a password sees a Job, gets a file or a link, or makes the Website call AWS on their behalf.

**Assumptions to confirm at review:** F (criteria 1, 2 and 3), L (criteria 3, 4 and 5). **Settled at review:** R7 (criterion 4).

#### Acceptance Criteria

1. WHILE a request has no Valid_Session, THE Website SHALL answer only the Open_Routes: GET of the sign-in page, POST of the sign-in form, GET of the stylesheet and GET of the favicon (assumption F).
2. WHILE a request has no Valid_Session, THE Website SHALL answer every request outside the Open_Routes, whatever its path, method, cookies or body size, the Showcase and the Page_Script included, with a redirect to the sign-in page, or with HTTP 401 when the request is a Script_Request (assumption F).
3. WHILE a request outside the Open_Routes has no Valid_Session, THE Website SHALL make no Presigned_Link and no AWS call for the request other than the one read of the Website_Table that checks a well-formed Session_Cookie (assumptions F and L).
4. IF a request's Session_Cookie doesn't hold an Account name that matches the Account_Name_Pattern and a well-formed Session id, THEN THE Website SHALL treat the request as signed out and make no AWS call for the Session_Cookie (assumption L, R7).
5. FOR ALL paths, methods, cookies and body sizes of requests outside the Open_Routes that have no Valid_Session, THE Website SHALL answer with the redirect to the sign-in page or with HTTP 401, and SHALL make no AWS call other than the read of criterion 3 (property), which the Test_Suite checks with the Fake_AWS recording every call, Presigned_Links included (assumption L).

### Requirement 6: The pages and their safety

**User Story:** As a judge on a phone or a laptop, I want plain pages that work in the browser I already have, so that I can upload, start and follow a Job from either, safely.

**Assumptions to confirm at review:** A (criteria 1 and 2).

#### Acceptance Criteria

1. THE Website SHALL render its pages on the server, and THE Page_Script SHALL be plain JavaScript that the Website serves with no build step, only for the part upload and the status refresh, so the Website needs no Node build (assumption A).
2. THE Website's pages SHALL set a viewport for phones and lay out in one column that fits a screen 360 CSS pixels wide, so the same pages work on a phone and a laptop (assumption A).
3. THE Website SHALL escape as HTML text every value it puts into a page, the titles, credits, file names, Account names and Readable_Errors included, so no value can add markup or script to a page.
4. THE Website SHALL send with every response a Content-Security-Policy that lets a page load scripts, styles and images only from the Website, send network requests only to the Website and the Job_Bucket, post forms only to the Website, and be framed by no page, together with `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer`.
5. THE Website SHALL send `Cache-Control: no-store` with every signed-in page and every answer to a Script_Request, so a shared browser keeps no copy of an Account's pages.
6. IF a request with a Valid_Session, or a request to an Open_Route, has a body larger than the longest request body in `[website]`, THEN THE Website SHALL answer HTTP 413 without reading the body further, because no request to the Website needs more and the video never passes through the Website_Function, and SHALL check the Session before the body's size, so a request outside the Open_Routes with no Valid_Session gets Requirement 5's redirect or HTTP 401 whatever its body's size.
7. WHEN a signed-in Account asks for a path the Website doesn't have, THE Website SHALL answer HTTP 404 with a page that links to the job list.
8. IF the Website meets an error it didn't foresee, THEN THE Website SHALL answer HTTP 500 with a page that says something went wrong and to try again, and SHALL log the error's type and its Account_Redacted message, showing no traceback, AWS error or ARN to the browser.
9. THE Website SHALL write no password, Session id, CSRF_Token, cookie or Presigned_Link to its log, and SHALL apply the Account_Redaction to every line it logs and to every error text it shows.

### Requirement 7: Uploading the video

**User Story:** As a judge with a 4 GB phone video, I want my browser to send the video straight to storage in parts, so that a big video uploads even though the Website_Function takes at most 6 MB per request.

**Settled in clarification:** Q7 (criteria 4, 5, 6 and 9), Q10 (criterion 10). **Assumptions to confirm at review:** A (criterion 1), K (criterion 2).

#### Acceptance Criteria

1. WHEN a signed-in Account opens the start page, THE Website SHALL offer a file picker for one video and the Presets fast, default and quality, with the Preset in `[jobs].default_preset` (fast) chosen when the page opens and each Preset shown with its Expected_Time from `[cloud.presets]` (assumption A).
2. WHEN a signed-in Account presses Start, THE Website SHALL run the Start_Precheck before the Upload's first part goes, and SHALL show the refusal of Requirement 11 in place of uploading when a check fails (assumption K).
3. WHEN a signed-in Account begins an Upload, THE Website SHALL refuse a file whose name doesn't end in `.mp4` or `.mov`, in any letter case, or whose stated size is over `[jobs].max_video_bytes`, with a Readable_Error that says what to change, as the Job_Maker does.
4. WHEN a signed-in Account begins an Upload, THE Website SHALL start an S3 multipart upload under `uploads/` in the Job_Bucket, at a key made from a new random id and never from the file's name, and SHALL record the Upload with its Account, its stated size and its file name in the Website_Table (Q7).
5. THE Page_Script SHALL send the video straight to the Job_Bucket in parts of the size in `[website]`, each part with its own Upload_Link, so the video never passes through the Website_Function (Q7).
6. THE Website SHALL make an Upload_Link only for a part of an unfinished Upload that the same Account began, and only for a part number that the Upload's stated size needs, valid for `[website].upload_link_seconds` (15 minutes) (Q7).
7. WHEN an Upload_Link expires before its part is sent, THE Page_Script SHALL ask the Website for a new Upload_Link for that part.
8. WHILE an Upload's parts are being sent, THE Page_Script SHALL show how much of the video has been sent.
9. WHEN the Page_Script has sent every part, THE Website SHALL complete the multipart upload, and SHALL refuse the Upload when the object's size in the Job_Bucket is over `[jobs].max_video_bytes` or differs from the stated size (Q7).
10. THE Job_Bucket SHALL delete each object under `uploads/` 1 day after the object was made, and abort an unfinished upload there 1 day after the upload began, as S3 counts lifecycle days, while a started Job's copy under `jobs/<Job_Id>/input/` stays like any Job's files (Q10).

### Requirement 8: Checking the video

**User Story:** As the person paying, I want the Website to check an uploaded video the way the Job_Maker would, without ffprobe and without fetching the whole video, so that a bad video fails for free before any machine starts.

**Settled in clarification:** Q4 (criteria 1, 2, 3, 5 and 8). **Assumptions to confirm at review:** B (criterion 7).

#### Acceptance Criteria

1. WHEN the Website_Caller starts a Job from a completed Upload, THE Video_Check SHALL check the Upload with the Box_Reader, over S3 range reads of the uploaded object, with no ffprobe, no new package and no fetch of the whole video (Q4).
2. THE Box_Reader SHALL walk the top-level Boxes from the start of the file, skipping each Box by its size, reading the 64-bit `largesize` when a Box's size is 1 and taking a Box whose size is 0 to run to the end of the file, until it finds the `moov` Box, wherever in the file the `moov` Box is (Q4).
3. WHEN the Box_Reader finds the `moov` Box, THE Box_Reader SHALL read the `moov` Box and find whether it holds a video track (a `trak` Box whose handler type is `vide`), and the Video_Length: the first video track's duration over its time scale from its media header, or the movie header's duration over its time scale when that track's duration is 0, as the pipeline-core spec defines the Video_Length (Q4).
4. THE Box_Reader SHALL read only each top-level Box's header and the `moov` Box, and never the media data.
5. IF the walk passes the most Boxes in `[website]`, or the `moov` Box is larger than the largest `moov` in `[website]`, THEN THE Video_Check SHALL refuse the Upload with a Readable_Error before reading further (Q4).
6. IF a Box's size is smaller than the Box's own header or runs past the end of the file, or the walk reaches the end of the file without a `moov` Box, THEN THE Video_Check SHALL refuse the Upload with a Readable_Error that says the file isn't an MP4 or MOV video the Pipeline can read and to save it again as MP4.
7. IF the `moov` Box holds no video track, or the Video_Length is 0 or can't be read, or the Video_Length is over `[jobs].max_video_seconds`, or the whole video is shorter than `[jobs].min_used_seconds` by `splatpipe.make.used_part` with no Trim, THEN THE Video_Check SHALL refuse the Upload with a Readable_Error that means the same as the Job_Maker's refusal in that case (assumption B).
8. FOR ALL layouts of top-level Boxes, with any types and sizes, `largesize` Boxes and an optional size-0 last Box, that hold one `moov` Box anywhere, THE Box_Reader SHALL find that `moov` Box and read the video track and the Video_Length that the layout was built with, and FOR ALL layouts with a broken size or no `moov` Box, THE Video_Check SHALL refuse the Upload (property) (Q4).

### Requirement 9: Starting a Job through the Start_Core

**User Story:** As the person paying, I want the Website to start each Job through the same code as `cloud.start`, so that the Smoke_Test gate, the lock-hash check, the G_Quota check and the Spend_Guard apply to every Website_Job.

**Settled in clarification:** Q14 (criteria 1 to 5). **Assumptions to confirm at review:** B (criterion 7). **Settled at review:** R7 (criterion 8).

#### Acceptance Criteria

1. THE Cloud_Jobs code SHALL split the Start_Command's `start_job` into the Start_Core and two thin callers, the Laptop_Caller and the Website_Caller, and THE Start_Core SHALL keep today's order: the Step_List check, the newest available Machine_Image, the lock-hash check, the Smoke_Marker check, the caller's Job_Record, the machine choice against the G_Quota, the money quote, the Spend_Guard, the uploads and the fallback launch (Q14).
2. THE Laptop_Caller SHALL keep today's uncommitted-change check, the Code_Archive and the lock hashes of `HEAD`, and `make_job`, and THE Start_Command SHALL keep its signature, its messages and its order, so every cloud-jobs test passes unchanged (Q14).
3. THE Website_Caller SHALL run the Baked_Commit: the Website_Caller SHALL upload the Baked_Commit's Code_Archive and give the Start_Core the Baked_Commit's `uv.lock` and `package-lock.json` hashes to compare with the Machine_Image's tags, so after a new Machine_Image with a different `uv.lock` every Website start refuses until a redeploy (Q14).
4. THE Website_Caller SHALL build the Job_Record from the Pipeline's public pieces, with no change to `splatpipe/`: a new Job_Id from `splatpipe.jobs.new_job_id`, the Preset through `splatpipe.config.check_preset`, the Used_Part through `splatpipe.make.used_part` with no Trim, and `splatpipe.jobs.write_record` (Q14).
5. THE Website_Caller SHALL copy the Upload to the Job_Record's `video_file` under `jobs/<Job_Id>/input/` inside S3, so the video passes through neither the Website_Function's disk nor its memory (Q14).
6. THE Website_Caller SHALL record the Upload's file name as the Job_Record's `source_name` only after dropping any folder part and any control character from the name and cutting the name to the longest file name in `[website]`.
7. THE Website SHALL use the whole video for every Website_Job, with no Trim on the Website, while the Start_Command keeps `--start` and `--end` (assumption B).
8. THE Website_Caller SHALL start a Job only from a completed Upload that the same Account began, only with one of the Presets fast, default and quality, and at most one Website_Job from each Upload, so a second press of Start can't start a second Job (R7).
9. IF the Upload is no longer in the Job_Bucket when the Account starts the Job, THEN THE Website_Caller SHALL refuse with a Readable_Error that says to upload the video again.
10. WHEN the Start_Core has launched a Website_Job's machine, or a launch error leaves it unclear whether a machine started, THE Website SHALL have recorded in the Website_Table the Job_Owner, the Preset, the start time and the number of Steps in the Baked_Commit's Step_List, so the job list shows the Job.
11. THE Start_Core's launch function SHALL call the Spend_Guard itself, for both the Laptop_Caller and the Website_Caller, before any upload, any copy inside S3 and any launch, and THE Website_Caller SHALL take the Job's Reservation just before calling the Start_Core's launch function.

### Requirement 10: The Website_Limits and Reservations

**User Story:** As the Account_Owner, I want a hard daily and total limit on the GPU machines the Website starts, approved once, so that judges start Jobs without asking me and the spend can never pass what I approved.

**Settled in clarification:** Q2 (criteria 12 and 13), Q3 (criteria 3 and 13), Q5 (criteria 1, 2, 4, 5, 6, 7, 8 and 10), Q10 (criterion 10), Q12 (criteria 1 and 11). **Assumptions to confirm at review:** M (criterion 7). **Settled at review:** R1 (criteria 2, 3 and 13), R7 (criterion 7), D2 (criterion 14).

#### Acceptance Criteria

1. WHEN the Website_Caller has chosen the machine and quoted the Job, THE Website SHALL take the Job's Reservation just before the Start_Core's launch function calls the Spend_Guard, so the Spend_Guard runs only after the Reservation is taken (Q5, Q12).
2. THE Reservation SHALL hold the Preset's Worst_Case_Cost in whole cents, 515 for fast, 762 for default and 1841 for quality with today's Config, which is never less than the worst case the Start_Command's quote gives for the same Preset, because the Start_Command launches only types on that Fallback_List, and the Worst_Case_Cost adds the boot margin and rounds up (Q5, R1).
3. THE Website SHALL take a Reservation only when, with the Reservation counted, the UTC_Day holds at most 10 Reservations and at most $52 of Reservations, and all Reservations together come to at most $500 (Q3, R1).
4. THE Website SHALL take each Reservation in one atomic conditional write to the Website_Table, so two starts at the same moment can't both take the last place (Q5).
5. THE Website SHALL count each Reservation in the UTC_Day of a clock read taken right before the Reservation's transaction, not when the request arrived, and SHALL give a Reservation back to that same UTC_Day (Q5).
6. THE Website SHALL never settle a Reservation down to the Job's real cost, so the Website_Limits hold even when a Job runs to its Time_Cap (Q5).
7. IF a Website start fails after its Reservation and before any machine could have started, because the Spend_Guard refuses, an upload or the copy fails, or AWS refuses every launch attempt for capacity, for the G_Quota or for EC2's instance limit, THEN THE Website SHALL give the Reservation back (Q5, assumption M, R7).
8. IF a launch fails with any other error, which might have started a machine, THEN THE Website SHALL keep the Reservation (Q5).
9. THE Website SHALL give a Reservation back at most once, so a repeated give-back changes no Limit_Counter.
10. THE Website SHALL count only the GPU machines it starts against the Website_Limits, and not storage, uploads or downloads (Q5, Q10).
11. THE Limit_Counters SHALL live in the Website_Table and SHALL outlast every redeploy, so a redeploy resets no Limit_Counter (Q12).
12. IF a clock read taken right before the Reservation's transaction, not when the request arrived, is at or after the Stop_Time, THEN THE Website SHALL take no Reservation (Q2).
13. FOR ALL interleavings of any number of starts at once, with refusals, give-backs, launch failures and UTC_Day changes between their steps, THE Limit_Counters SHALL never show a UTC_Day with more than 10 Reservations or more than $52 of Reservations, or more than $500 of Reservations in total, SHALL never go below 0, and SHALL never gain a Reservation at or after the Stop_Time (property) (Q2, Q3, R1).
14. IF a clock read taken right before the launch, after the copy inside S3 and the uploads, is at or after the Stop_Time, THEN THE Website SHALL give the Reservation back, launch no machine and show the Stop_Time message of Requirement 11's criterion 3 (D2).

### Requirement 11: When a Job can't start

**User Story:** As a judge, I want a plain message when my Job can't start that says whether to try again later, so that I'm never left guessing and nothing waits in a queue.

**Settled in clarification:** Q2 (criterion 3), Q3 (criterion 2), Q5 (criterion 1). **Assumptions to confirm at review:** D (criteria 1 and 4), I (criterion 3), M (criterion 1). **Settled at review:** R7 (criterion 1).

#### Acceptance Criteria

1. IF `admit` refuses the start for the G_Quota, or AWS refuses every launch attempt for capacity, for the G_Quota or for EC2's instance limit, THEN THE Website SHALL show "All GPU machines are busy right now, so your job didn't start. Try again later.", give back any Reservation and queue nothing (Q5, assumptions D and M, R7).
2. IF a Website_Limit has no room for the Job's Reservation, THEN THE Website SHALL show a Readable_Error that names the limit and says no Job started, and for the Daily_Job_Limit or the Daily_Cost_Limit, that Jobs can start again after 00:00 UTC (Q3).
3. WHILE the Stop_Time has passed, THE Website SHALL show on the start page that the Website stopped starting Jobs at the Stop_Time, written from Config as 08:00 UTC on 4 February 2027, SHALL refuse to begin an Upload or start a Job, and SHALL keep every other page working (Q2, assumption I).
4. IF a start fails for a reason a judge can't fix (no available Machine_Image, no Smoke_Marker, a lock-hash drift, a G_Quota of 0, an empty Step_List, Step_Needs that no type fits, or a missing Stack value), THEN THE Website SHALL show "The website can't start jobs right now." and log the reason, Account_Redacted (assumption D).
5. IF the Video_Check refuses the Upload, THEN THE Website SHALL show the Video_Check's Readable_Error.
6. IF a launch fails with an error that might have started a machine, THEN THE Website SHALL say that the Job may or may not have started and that its Website_Status will show which, and SHALL keep the Job in the Account's job list.
7. THE Website SHALL never retry, relaunch or queue a start by itself after a refusal or a failure, so each start launches at most one machine, as the Rules_Steering says.

### Requirement 12: The job list, the statuses and the Result_Page

**User Story:** As a judge, I want to see my Jobs with their status and, for a finished Job, its result and one download, so that I get the splat without asking anyone.

**Settled in clarification:** Q7 (criteria 9 and 10), Q8 (criterion 3). **Assumptions to confirm at review:** C (criteria 3 and 4), G (criteria 1 and 2), H (criteria 5, 7, 8 and 11). **Settled at review:** R4 (criteria 3, 14 and 15).

#### Acceptance Criteria

1. WHEN a signed-in Account opens the job list, THE Website SHALL list the Website_Jobs that the Account started and whose machine launched or might have launched, newest first, up to the most Jobs in `[website]`, each with its Website_Status and a link to its Result_Page (assumption G).
2. THE Website SHALL show an Account only the Website_Jobs that the Account started, and Jobs from a laptop or from another Account only through the Showcase (assumption G).
3. THE Website SHALL work out each Website_Status from the Job's files in the Job_Bucket: Starting while the Job's `out/` holds no file; Running (step n of N) while `out/` holds files but no Job_Summary, where n is the number of Step_Reports plus 1, at most N, and N is the number of Steps the Job runs; Done, Done with warnings or Failed from the Job_Summary's status, with Failed showing the Job_Summary's Readable_Error; and Ended without a result, in place of Starting or Running, only as criterion 14 says (Q8, assumption C, R4).
4. WHILE a Result_Page shows Starting or Running, THE Page_Script SHALL refresh the Website_Status every `[cloud].poll_seconds` seconds (assumption C).
5. WHEN a signed-in Account opens the Result_Page of a Job that the Account may see, a Website_Job the Account started or a Showcase_Entry, THE Website SHALL show the Website_Status, the Preset, when the Job started and, once the Job has them, how long the run and each Step took (assumption H).
6. WHILE a Job's Website_Status is Done with warnings, THE Result_Page SHALL show the Readable_Error of each Optional_Step that failed, so a judge sees why the download is the trained splat.
7. WHILE a Job's Website_Status is Done or Done with warnings, THE Result_Page SHALL offer one download button, for `out/clean.sog` (the name in `[clean].clean_sog`), or for `out/train.ply` (the name in `[clean].train_ply`) when the Job's `out/` holds no `clean.sog` (assumption H).
8. IF the `out/` of a Job that is Done or Done with warnings holds neither file, THEN THE Result_Page SHALL say that the Job has no splat to download and offer no download button (assumption H).
9. WHEN a signed-in Account presses the download button, THE Website SHALL check that the Account may see the Job, then answer with a redirect to a Download_Link for that one Result_File, valid for `[website].download_link_seconds` (5 minutes) (Q7).
10. THE Website SHALL make a Download_Link only for a Job's Result_File, and for no other key in the Job_Bucket (Q7).
11. THE Result_Page SHALL hold one empty, labelled Viewer_Slot where the Viewer_Spec will put the 3D viewer, and THE Website SHALL put no viewer in the Viewer_Slot in this spec (assumption H).
12. IF a signed-in Account asks for a Job that the Account may not see, THEN THE Website SHALL answer HTTP 404, the same as for a Job that doesn't exist, so the answer doesn't show which Job_Ids exist.
13. THE Website SHALL check every Job_Id it receives with `splatpipe.jobs.check_job_id` before using the Job_Id in a key.
14. WHILE a Job has no Job_Summary, THE Website SHALL show Ended without a result only when all of these hold: at least `[website].ended_check_minutes` (10 minutes) have passed since the Job started; a lookup of the machines that carry the Job's Job_Id tag and the Project_Tag finds each of them terminated, or finds none; and a second read of the Job's `out/job.json` after that lookup still finds no Job_Summary (R4).
15. IF the machine lookup of criterion 14 fails, THEN THE Website SHALL keep the Job's Starting or Running status (R4).

### Requirement 13: The Showcase

**User Story:** As the Account_Owner, I want to pick finished Jobs for a Showcase, each with a title and its footage's credit, so that a judge sees results without waiting and every credit the footage needs is shown.

**Settled in clarification:** Q11 (criterion 9), Q13 (criteria 1 to 7).

#### Acceptance Criteria

1. WHEN the Account_Owner adds a Job to the Showcase with the Showcase_Command, THE Showcase_Command SHALL check that the Job's Job_Summary in the Job_Bucket says done or done with warnings and that the Job's `out/` holds a Result_File, and SHALL refuse with a Readable_Error otherwise (Q13).
2. THE Showcase_Command SHALL take any such Job, whether the Website or the Start_Command started it (Q13).
3. THE Showcase_Command SHALL take a title for each Showcase_Entry, at most as long as the longest title in `[website]`, and the Showcase_Entry's Footage_Licence (Q13).
4. IF the Footage_Licence is CC BY or CC BY-SA and the Account_Owner gives no author's credit, THEN THE Showcase_Command SHALL refuse the Showcase_Entry with a Readable_Error that says the licence needs the author's credit (Q13, Rules_Steering).
5. WHEN the Showcase_Command adds or removes a Showcase_Entry, THE Showcase SHALL show the change on its next load, with no redeploy (Q13).
6. WHEN a signed-in Account opens the Showcase, THE Website SHALL list every Showcase_Entry, newest first, with its title and a link to its Result_Page, and beside each entry made from CC BY or CC BY-SA footage, the author's credit and the licence, adding for CC BY-SA footage that the splat is shared under CC BY-SA 4.0 (Q13, Rules_Steering).
7. THE Result_Page of a Showcase_Entry SHALL show the same title, credit and licence beside the result (Q13).
8. THE Showcase_Command SHALL list the Showcase_Entries and remove one Showcase_Entry by its Job_Id.
9. THE Showcase_Command SHALL need only the Python standard library and boto3, and SHALL run only in CloudShell, as Requirement 1 says (Q11).

### Requirement 14: The Stack changes

**User Story:** As the Account_Owner, I want the Website's AWS parts added to the one template with the least rights the Website needs, so that I apply them with my usual change set and the Website can do no more than start and read Jobs.

**Settled in clarification:** Q6 (criteria 1 and 2), Q7 (criterion 7), Q8 (criteria 4 and 12), Q9 (criteria 1, 8, 9, 10, 11 and 13), Q10 (criteria 5 and 6), Q11 (criterion 12), Q12 (criterion 14). **Settled at review:** R2 (criteria 1, 9, 13 and 17), R3 (criterion 4).

#### Acceptance Criteria

1. THE Stack_Template SHALL stay the one template, and SHALL gain the Website_Function on Lambda's Python 3.11 runtime, the Website_URL, the Website_Function's log group, the Website_Table, the Website_Role with a managed policy, one CORS rule and one lifecycle rule on the Job_Bucket, and the Website_Code_Parameters, with no API Gateway, CloudFront, App Runner, load balancer or EC2 web server (Q6, Q9, R2).
2. THE Website_URL SHALL use the `NONE` auth type, and THE Stack_Template SHALL give the Website_Function a resource policy that lets anyone call `lambda:InvokeFunctionUrl` for the `NONE` auth type and `lambda:InvokeFunction` only through the function URL (`lambda:InvokedViaFunctionUrl`), both of which a function URL made since October 2025 needs, so the Website does its own sign-in (Q6).
3. THE Stack SHALL give the Website_URL as a Stack_Output, so the Account_Owner can hand the Website_URL to the judges.
4. THE Website_Table SHALL use on-demand billing, a time-to-live attribute for its Sessions, its Sign_In_Failures and its Uploads, and point-in-time recovery, and THE Stack_Template SHALL give the Website_Table the Job_Bucket's `DeletionPolicy: RetainExceptOnCreate` and `UpdateReplacePolicy: Retain`, so deleting the Stack, or an update that replaces the Website_Table, keeps the Accounts, the Limit_Counters and the Showcase_Entries (Q8, R3).
5. THE Website_Function's log group SHALL keep its logs for 30 days (Q10).
6. THE Job_Bucket's new lifecycle rule SHALL delete each object under `uploads/` 1 day after the object was made and abort an unfinished upload under `uploads/` 1 day after the upload began, and SHALL reach nothing outside `uploads/` (Q10).
7. THE Job_Bucket's CORS rule SHALL allow only the PUT method, only from the Website_URL's origin, and SHALL expose only the `ETag` header to the Page_Script (Q7).
8. THE Website_Role SHALL launch Job_Machines only from a Machine_Image, owned by the account and carrying the Project_Tag, under every condition of the Runner_Role's launch policy (the Allowed_Types, IMDSv2, the Project_Tag on the request, the Job_Profile, the Job_Security_Group, a subnet of the Default_VPC, tags only when launching, and no key pair or launch template), and SHALL pass only the Job_Role, and only to EC2 (Q9).
9. THE Website_Role SHALL read the G_Quota, describe EC2 in the Stack_Region, list and read only `jobs/` and `uploads/` in the Job_Bucket, and call `cloudformation:DescribeStacks` on its own Stack only, which the Website uses to find its resources at run time (Q9, R2).
10. THE Website_Role SHALL write to the Job_Bucket only under `jobs/<Job_Id>/input/` and `uploads/`, with resource patterns that can't reach a Job's `out/` or the Bucket_Work_Folders (an IAM `*` also matches `/`, so the pattern pins the Job_Id's shape), and SHALL use only the Website_Table and the Website_Function's own log group (Q9).
11. THE Website_Role SHALL have no action that terminates or stops a machine, makes or removes an image or a snapshot, opens a Debug_Shell, deletes an S3 object, a Website_Table item or a log, or reads or changes IAM, and SHALL reach no Other_Resource (Q9).
12. THE Stack_Template SHALL leave the Read_Policy, the Runner_Role and its policies, the Job_Role and its policy and the Project_Group exactly as they are, so the Read_Profile and the Runner_Profile keep exactly the rights they have now and gain no right on the Website_Table (Q8, Q11).
13. WHERE the Website_Code_Parameters are empty, THE Stack SHALL still create, with the Placeholder_Function as the Website_Function's code, so a fresh copy's first create works without a Website_Bundle (Q9, R2).
14. THE Website_Function's environment SHALL hold no Spend_Variable and no secret (Q12).
15. THE Stack_Template SHALL put the Project_Tag on each new resource whose CloudFormation type takes tags (the Website_Function, its log group, the Website_Table and the Website_Role), and SHALL stay within 51,200 bytes, the most that `--template-body` takes, as the aws-setup spec requires.
16. THE Stack_Template SHALL pass the Template_Test with no cfn-lint error or warning, and SHALL pass the Compliance_Check with each failing rule fixed, or suppressed as an Unfit_Rule with a one-line reason the aws-setup spec's way, counting as Unfit_Rules also the rule against public access to a function (the Website_URL has to answer browsers) and the rules that ask a function for a VPC, a dead-letter queue or reserved concurrency; IF another rule fails that the Builder can't fix, THEN THE Builder SHALL ask the user before suppressing it.
17. THE Stack_Template SHALL let `WebsiteCodeKey` be only empty or `website/` followed by 40 hexadecimal characters and `.zip` (R2).

Note: a CORS rule on the Job_Bucket that names the Website_URL's origin, together with a Website_Function that is given the Job_Bucket's name, would make those resources depend on each other in the Stack_Template. The design must break that loop without widening the CORS rule, or raise it at review. Settled at review by R2: the Website_Function names no Job_Bucket in the Stack_Template, because its code comes from the Website_Code_Parameters and the Website reads the Job_Bucket's name from the Stack at run time.

### Requirement 15: The Deploy_Command

**User Story:** As the Account_Owner, I want one command that builds and uploads the Website from a clean commit inside CloudShell's small space and stops for my review, so that a deploy fits CloudShell, runs committed code and carries my one yes.

**Settled in clarification:** Q9 (criterion 8), Q11 (criteria 1, 2 and 3), Q12 (criteria 7 and 9), Q14 (criteria 4 and 5). **Settled at review:** R1 (criterion 7), R2 (criteria 8, 12 and 13), R7 (criterion 7).

#### Acceptance Criteria

1. THE Deploy_Command SHALL run only in CloudShell, as Requirement 1 says, from a clone of the repo under `/tmp`, and only the Account_Owner SHALL run it (Q11).
2. THE Deploy_Command SHALL keep its clone, its build, its caches and the Website_Bundle under the folder in `/tmp` that `[website]` names, so nothing the Deploy_Command makes goes in CloudShell's 1 GB home folder (Q11).
3. THE Deploy_Command SHALL install only what the deploy needs, Python 3.11 and boto3 at `app/uv.lock`'s versions, and SHALL never run a `uv sync` of the root project, which would install numpy, scipy, OpenCV and the dev tools (Q11).
4. IF `app/`, `splatpipe/`, `cloud/` or `config.toml` has uncommitted changes, THEN THE Deploy_Command SHALL stop with a Readable_Error that names the changed paths, before it builds anything, as the Start_Command does (Q14).
5. THE Deploy_Command SHALL make the Code_Archive with `git archive HEAD` and the `uv.lock` and `package-lock.json` hashes from `HEAD`'s committed files, and SHALL put them and `HEAD`'s commit id in the Website_Bundle as the Baked_Commit (Q14).
6. THE Website_Bundle SHALL hold only what the Website_Function runs: `app/`, the `cloud/` and `splatpipe/` modules the Website imports, `config.toml`, the Baked_Commit, and boto3 with the packages boto3 pulls in at `app/uv.lock`'s versions, with no test or doc outside the Baked_Commit's Code_Archive, and no video or GPU package.
7. WHEN the Deploy_Command has built the Website_Bundle, THE Deploy_Command SHALL call the Spend_Guard with the Website_Limits' worst case, written from Config and split across `check_spend`'s three fields: `machine` "website GPU jobs (up to 10 and $52 a UTC day)", `expected_time` "every day until 2027-02-04 08:00 UTC" and `rough_cost` "at most $500 in total", before the Deploy_Command uploads anything or makes the change set, so `APPROVED_SPEND=1` on the Deploy_Command is the Account_Owner's one yes for the Website_Limits (Q12, R1, R7).
8. WHEN the Spend_Guard returns, THE Deploy_Command SHALL upload the Website_Bundle to the Job_Bucket at `website/<commit id>.zip`, under a prefix of its own that no expiration rule reaches and the Website_Role can't write, where the commit id is the Baked_Commit's; then make, and not execute, a change set of the Stack from the same commit's Stack_Template, sent inline as the change set's template body and never as a template URL, with the Website_Code_Parameters set to the Stack's `BucketName` output and the bundle's key, the DefaultVpcId's previous value and the Stack's Project_Tag; and print the change set's review lines (each change's action, logical id and replacement, with no physical id), for the Account_Owner to review and execute with the Setup_Guide's commands (Q9, R2).
9. THE Deploy_Command SHALL read and write nothing in the Website_Table, so a redeploy resets no Limit_Counter, Account or Showcase_Entry (Q12).
10. THE Deploy_Command SHALL use no interactive prompt, and SHALL print no account id, ARN or password, applying the Account_Redaction to every AWS error it prints.
11. THE Deploy_Command SHALL take its environment, its AWS client and its repo folder from its caller, so the Test_Suite runs the Deploy_Command against the Fake_AWS and a temporary git repo, with no CloudShell, no account and no network.
12. THE Deploy_Command SHALL fill `WebsiteCodeBucket` only with the Stack's own `BucketName` output and `WebsiteCodeKey` only as `website/<commit id>.zip`, with the Baked_Commit's 40-character commit id, and SHALL refuse any other value with a Readable_Error before it uploads anything (R2).
13. IF the Baked_Commit's Stack_Template is larger than 51,200 bytes, THEN THE Deploy_Command SHALL stop with a Readable_Error before it uploads anything, because the change set takes the Stack_Template inline (R2).
14. IF, after the first website deploy, the Deploy_Command's change set would remove, add or replace the Website_Table, or remove or replace the Website_Function or the Website_URL, THEN THE Deploy_Command SHALL delete that change set and stop with a Readable_Error that names the change, because a new Website_Table would start the Limit_Counters at 0 and the Website_URL is the judges' address.

### Requirement 16: Money and the one yes

**User Story:** As the Account_Owner, I want my one yes on the deploy to cover the Website's Jobs within the Website_Limits, and the Website to stay nearly free while nobody uses it, so that judges never wait for me and I never pay for an idle server.

**Settled in clarification:** Q2 (criterion 7), Q6 (criteria 5 and 6), Q12 (criteria 1, 2 and 3). **Assumptions to confirm at review:** I (criterion 7). **Settled at review:** R8 (criterion 1).

#### Acceptance Criteria

1. THE Account_Owner's `APPROVED_SPEND=1` on the Deploy_Command SHALL be the one yes for every Website_Job within the Website_Limits until the Stop_Time, whoever starts the Website_Job, the team included, in place of a yes per Job, while a Job started by hand with the Start_Command still needs its own yes (Q12, R8).
2. WHEN a Website_Limit or the Stop_Time changes in Config, THE change SHALL take effect only through a redeploy, whose Deploy_Command asks for a new yes through the Spend_Guard (Q12).
3. THE Website SHALL never read the Spend_Variable from the Website_Function's environment, and WHEN a Reservation has been taken, THE Website_Caller SHALL give the Start_Core's launch function, for its Spend_Guard call, an environment built in code that sets the Spend_Variable to `1` for that one call (Q12).
4. FOR ALL Website starts, Presets and Uploads, no upload, no copy inside S3 and no launch SHALL happen unless the Reservation has been taken and then the Spend_Guard has returned without raising (property), which the Test_Suite checks with the Fake_AWS's ordered call list.
5. THE Website SHALL run as one Lambda function with a function URL and one on-demand table, with no server, no domain and no resource that bills by the hour, so the Website costs about nothing while nobody uses it (Q6).
6. THE Website_Function SHALL use Lambda's Python 3.11 runtime, which Lambda supports until 30 June 2027, after the Stop_Time (Q6).
7. THE Website SHALL keep working, free for the judges, at least until the Stop_Time, and WHILE the Stop_Time has passed, THE Website SHALL keep everything except starting Jobs working until the Account_Owner takes the Website down (Q2, assumption I).

### Requirement 17: Docs, steering and the skill

**User Story:** As one of the people building the steps, I want the Website written up where the rest is and the rules brought up to date, so that the deploy, the Accounts and the money rules are clear to the Account_Owner, to the team and to Kiro.

**Settled in clarification:** Q3 (criterion 4), Q9 (criteria 1 and 2), Q10 (criterion 1), Q11 (criteria 1 and 5), Q12 (criteria 4 and 9), Q15 (criterion 10). **Assumptions to confirm at review:** H (criterion 13), I (criteria 1 and 8). **Settled at review:** R1 (criterion 4), R2 (criteria 2 and 16), R3 (criterion 3), R8 (criteria 4 and 9), R9 (criterion 1), R10 (criterion 1), R12 (criterion 17).

#### Acceptance Criteria

1. THE Setup_Guide SHALL gain a section on the Website that gives: the CloudShell_Steps, which put a clone, Python 3.11 and boto3 under `/tmp`, check that `echo $AWS_EXECUTION_ENV` prints `CloudShell` before any CloudShell_Command runs, and run again in each CloudShell session because CloudShell clears everything outside the home folder when a session ends; deploying and updating the Website with the Deploy_Command and the change-set review; making the judges' shared Account with the Account_Command; adding and removing Showcase_Entries; the Website_Limits, the Stop_Time and the one yes; what the Website costs; that the Project_Group's Read_Policy, and so the Read_Profile that Kiro uses, can read `uploads/` and the Website_Jobs' inputs; clearing the Website_Jobs from the Job_Bucket after judging; and taking the Website down after the Stop_Time (Q9, Q10, Q11, assumption I, R9, R10).
2. THE Setup_Guide's change-set review SHALL check every stack update after the first website deploy, as the Deploy_Command does, for a change that removes, adds or replaces the Website_Table, which would give the Website a new, empty Website_Table with no Accounts or Showcase_Entries and its Limit_Counters at 0, or that removes or replaces the Website_Function or the Website_URL, the judges' address; and THE Setup_Guide's "Updating the stack" command SHALL keep the Website_Code_Parameters' previous values once the Website is deployed, so a later stack update doesn't swap the Website for the Placeholder_Function (Q9, R2).
3. THE Setup_Guide's "Removing the setup" SHALL say what happens to the Website_Function, the Website_Table, the Website_Function's log group and the Website_Bundles in the Job_Bucket, and that deleting the Stack keeps the Website_Table, as it keeps the Job_Bucket (R3).
4. THE Rules_Steering SHALL give the exception for Jobs started through the app its numbers (10 Jobs and $52 a UTC_Day, and $500 in total, until the Stop_Time, set in `[website]`), and SHALL say that `APPROVED_SPEND=1` on the Deploy_Command is the Account_Owner's one yes for them, that they count against the Website_Limits whoever starts them, the team included, while a Job started by hand with the Start_Command still needs its own yes, that a limit change needs a redeploy and a new yes, and that the Website never holds the Spend_Variable (Q3, Q12, R1, R8).
5. THE Rules_Steering SHALL say that the CloudShell_Commands are the Account_Owner's alone, run in CloudShell with the Admin_Login, and that Kiro never runs them (Q11).
6. THE Tech_Steering SHALL describe the Website: one Lambda function on Python 3.11 with a function URL and one on-demand DynamoDB table, the Python standard library for the web parts, AWS only through `cloud/`, the App_Project with only boto3, and the Start_Core shared with `cloud.start`.
7. THE Structure_Steering SHALL list `app/` as the Website's code, with the App_Project, and say that the Website's packages never go in the root Project_File or `uv.lock`.
8. THE Product_Steering SHALL replace "The app around the pipeline comes later" with a paragraph about the Website: signing in, uploading a video, picking a Preset, starting a Job, the job list, the Result_Page and its download, the Showcase, and the viewer to come (assumption I).
9. THE AWS_Jobs_Skill SHALL give the exception for Jobs started through the app the same numbers, the same one yes on the Deploy_Command and the same rule for Jobs that the team starts as the Rules_Steering, and SHALL say that Kiro never runs the CloudShell_Commands (Q12, R8).
10. THE Licence_List SHALL gain a row for anything the Website adds that the Licence_List doesn't list yet, among them Python (PSF-2.0), whose `python3.11` in CloudShell runs the CloudShell_Commands and whose Lambda runtime runs the Website, and pip (MIT), which installs the Website_Bundle's packages in CloudShell, and SHALL update the "How we use it" of each row whose use changes: boto3 and the packages boto3 pulls in, which the Website_Bundle now ships to the Website_Function, and uv, if the CloudShell_Steps use it (Q15).
11. THE README SHALL say that the Website exists, that the Setup_Guide documents its deploy, and that the CloudShell_Commands are the Account_Owner's, because the README names the commands.
12. THE Pipeline_Doc's "Running a job in the cloud" section SHALL say that the Website starts Jobs through the same Start_Core as `cloud.start`, and reads a Job's Website_Status from its Job_Summary and Step_Reports and its Result_File by the names in `[clean]`, so a Step spec that changes those files knows the Website reads them.
13. THE Pipeline_Doc SHALL say that the Viewer_Spec decides the splat-transform version the viewer is pinned to, which the Clean_Spec left to the app spec (assumption H).
14. THE Test_Suite SHALL check the lines this requirement adds to the docs, the steering and the skill, the way `tests/test_cloud_docs.py` checks the cloud-jobs spec's.
15. THE Cloud_Steering SHALL say that the Website takes its Reservation, a free write to the Website_Table, just before the Spend_Guard, which still comes before any upload, copy or launch.
16. THE Setup_Guide's commands SHALL type no bucket name, so the Website_Code_Parameters name only our own Job_Bucket: the Deploy_Command fills `WebsiteCodeBucket` from the Stack's `BucketName` output, and the Setup_Guide's other stack commands keep its previous value, or leave it empty before the first deploy (R2).
17. THE Rules_Steering and THE AWS_Jobs_Skill SHALL each hold the same line, saying that Kiro never opens or downloads anything under `uploads/` or under any Job's `input/` in the Job_Bucket, because those hold judges' and users' videos, and that Kiro never needs them, because the Status_Command and the Download_Command read only a Job's `out/` (R12).

### Requirement 18: Tests

**User Story:** As a team member, I want every part of the Website tested on a laptop CPU against a Fake_AWS, so that the tests are quick, free and never touch real AWS.

**Settled in clarification:** Q15 (criterion 1). **Settled at review:** R4 (criterion 6), R9 (criterion 16).

#### Acceptance Criteria

1. THE Test_Suite SHALL test the Website on a CPU in the root environment, against the Fake_AWS of `tests/fake_aws.py` extended with the Website's new operations, with no account, no network and no real AWS call (Q15).
2. THE Fake_AWS SHALL carry out the Website_Table's conditional writes faithfully: a write whose condition fails changes nothing and raises the condition failure, and a transaction applies all of its writes or none of them.
3. THE Fake_AWS SHALL record every new operation, each Presigned_Link included, in its ordered call list, so a test can assert what the Website called and in which order.
4. THE Test_Suite SHALL hold unit tests for signing in: the right password, a wrong password, an unknown name, the wait after too many Sign_In_Failures from one Sign_In_Address, a Session's expiry, the Session_Cookie's attributes, the CSRF_Token and signing out.
5. THE Test_Suite SHALL hold unit tests for the Website_Limits: each limit's refusal, the UTC_Day boundary, giving a Reservation back and keeping one, and the Stop_Time.
6. THE Test_Suite SHALL hold unit tests for the job list: which Jobs an Account sees, each Website_Status from the Job_Bucket's files, Ended without a result with each of its three conditions and with a failed machine lookup, and the choice of the Result_File (R4).
7. THE Test_Suite SHALL hold unit tests for starting a Job: the Website_Caller's order through the Start_Core, the Video_Check, the copy inside S3, the Job_Record from the Pipeline's public pieces, and the busy and can't-start messages.
8. THE Test_Suite SHALL hold the property test of Requirement 10, criterion 13, written with Hypothesis: for all interleavings of starts, refusals, give-backs, launch failures and UTC_Day changes, the Limit_Counters never pass a Website_Limit, never go below 0 and never gain a Reservation at or after the Stop_Time.
9. THE Test_Suite SHALL hold the property test of Requirement 5, criterion 5, written with Hypothesis: for all paths, methods, cookies and body sizes of signed-out requests outside the Open_Routes, the answer is the redirect or HTTP 401, and the Fake_AWS records no call but the one Session read.
10. THE Test_Suite SHALL hold the property tests of Requirement 8, criterion 8, and Requirement 16, criterion 4, written with Hypothesis.
11. THE Test_Suite's unit tests and property tests SHALL both be required, never optional, and each task SHALL write its own.
12. THE Test_Suite SHALL set the Spend_Variable in the process environment for no test, and SHALL make no AWS call, so running the Test_Suite spends nothing.
13. THE Test_Suite SHALL check that no file under `app/` imports boto3 or botocore, that no file under `splatpipe/` imports `app`, that the Project_File and the root `uv.lock` are unchanged, and that every package in `app/uv.lock` has the root `uv.lock`'s version, while the review checks with `git diff`, not with a test, that the cloud-jobs and pipeline-core test files stay unchanged.
14. THE Test_Suite SHALL check the Stack_Template's Website parts: the Website_Role's actions and resource patterns, the CORS rule, the lifecycle rule, the Website_Function's environment, the Project_Tags, and that the Read_Policy, the Runner_Role's policies and the Job_Role's policy are unchanged.
15. THE Test_Suite SHALL run each CloudShell_Command with an environment and a Fake_AWS that the test passes in, and the Deploy_Command also with a temporary git repo, and SHALL check each CloudShell_Command's refusal to run outside CloudShell.
16. THE Test_Suite SHALL check in `tests/test_guard.py` that the Guard_Hook blocks a shell command that runs each module named in `[website].cloudshell_modules` (R9).
