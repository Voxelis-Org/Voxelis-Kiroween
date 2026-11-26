---
name: aws-jobs
description: How our job machines run on EC2, with our money rules first. One On-Demand GPU machine per job, launched from our own machine image with the launch settings the runner role requires, the job bucket's jobs/ and work/ layout, user data, IMDSv2, terminating itself when the job ends, and Session Manager shells for debugging. Use when writing or reviewing the start, image-build, smoke-test or stop-all commands, a run_instances or create_image call or a job machine's user data script, when working out what a run costs, or when debugging a job machine.
metadata:
  sources: AWS docs for EC2, VPC, S3, Systems Manager and the Deep Learning AMIs, the boto3 reference, and the cloud-init and systemd docs, read November 2026
---

# Job machines on EC2

Notes for writing and reviewing the code that runs jobs on EC2. The money rules come first, because every machine costs money.

## Money rules

Copied from `.kiro/steering/rules.md`, section `## Money`. If the two ever differ, rules.md wins.

- The owner of the AWS account that the jobs run on approves people in advance, by adding them to the project's AWS group. That is their OK to run jobs on that account. After that, whoever is in the chat says yes to each paid run (cloud machines, cloud runs, deploys), once Kiro has told them the machine, the expected time and the rough cost.
- The exception is jobs that people start through the app (the judges now, other users later). The account owner approves those in advance, as a daily job limit and a daily cost limit that the app enforces. The numbers come in the app spec.
- Kiro and the team still need an OK for every run they start by hand.
- It runs on AWS, in the account of whoever deploys it (for our team, Samarth's).
- Speed and reliability matter more than saving a few cents, so pick the faster machine when it helps.
- Once the person in the chat says yes to a paid run, Kiro runs our start, image-build or smoke-test command itself, with `APPROVED_SPEND=1` on that one command.
- The code has a spend guard: nothing paid starts unless `APPROVED_SPEND=1` is set. Kiro never sets it otherwise.
- One machine per job at most. No automatic retries or relaunches.
- Nothing stays running after a job. Every machine shuts itself down when its job ends.

What they mean for job machines:

- Every launch is a paid run: a job, the smoke test and an image build alike. Apart from the jobs people start through the app, each launch needs its own yes, after Kiro has told the person the machine, the expected time and the rough cost.
- Every paid entry point, the start, image-build and smoke-test commands included, calls `check_spend()` from `cloud/spend.py` with that machine, time and cost, before it uploads or launches anything (cloud.md).
- When Kiro runs a raw `aws ec2 run-instances`, `aws ec2 create-image` or another command in `[paid].aws` in `config.toml`, the guard hook blocks it. So Kiro only starts paid things through our commands.
- One yes covers the whole fallback list. On a capacity error the command tries the type's other zones and then the next fitting type on the list, as part of the same approved run, because only one machine ever launches and nothing bills until one does. Any other launch error, or no capacity anywhere on the list, ends the command; starting the job again is a new run with a new yes.
- Nothing in our AWS setup stops a machine for us. The budget alert from Step 6 of the setup guide only sends an email. What stops the spend is the machine's own shutdown, its time cap and, as the last resort, the stop-all command.

### What a run costs

| What | How it's billed |
|---|---|
| The machine | Per second, with a one-minute minimum, only while it's `running`. Not while it's `pending`, `shutting-down` or `terminated` ([instance states](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-lifecycle.html)). |
| Its root EBS volume | For as long as the volume exists, whatever state the machine is in ([instance states](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-lifecycle.html)). Termination deletes it when its `DeleteOnTermination` is true, the default for a root volume attached at launch ([termination](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-ec2-instance-termination-works.html)). |
| Its public IPv4 address | $0.005 an hour, like every in-use public IPv4 address since February 2024 ([AWS News Blog](https://aws.amazon.com/blogs/aws/new-aws-public-ipv4-address-charge-public-ip-insights/)). A machine in a default subnet gets one by default ([default subnets](https://docs.aws.amazon.com/vpc/latest/userguide/default-subnet.html)) and uses it to reach S3 and Session Manager through the default VPC's internet gateway ([default VPCs](https://docs.aws.amazon.com/vpc/latest/userguide/default-vpc.html)). |
| Our machine image | Its snapshots, every month, until the image is deregistered and the snapshots deleted ([creating an AMI](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/creating-an-ami-ebs.html)). Deregistering alone leaves the snapshots, still billed ([deregistering](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/deregister-ami.html)). |
| The job bucket | Its files, by size, until someone deletes them or the 60-day rule deletes the work files. |

Instance metadata requests are free ([IMDS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html)), and so are Session Manager shells on EC2 ([AWS startup security baseline](https://docs.aws.amazon.com/prescriptive-guidance/latest/aws-startup-security-baseline/wkld-06.html)).

The hourly price of each allowed type goes in `config.toml`, next to the time caps (cloud.md). Neither is there yet. Take the prices for us-east-1 from the [EC2 On-Demand pricing page](https://aws.amazon.com/ec2/pricing/on-demand/). The rough cost to quote is the expected hours times the type's hourly price, plus $0.005 an hour for the address and the root volume's storage for the same hours. The time cap times the hourly price is the most a run costs, as long as the cap works.

## One machine per job

A job runs on one GPU machine, started from our own machine image. The machine gets its orders from its user data, reaches AWS through IMDSv2 and its instance role, writes its results to the job bucket and terminates itself when the job ends. While it runs, a person can open a shell on it with Session Manager.

None of this is built yet: the start, image-build, smoke-test and stop-all commands, the user data script and the machine image come with later specs. The AWS resources they use come from `cloud/aws-setup.yaml`, which `docs/aws-setup.md` explains. AWS's pages are paraphrased here, each with a link.

The plan, from the rules and the setup guide:

1. After the yes, the start command calls the spend guard, uploads the video and `job.json` to `jobs/<job id>/input/` with the runner profile, and launches one machine.
2. The machine boots our image, and cloud-init runs the user data script once, as root.
3. The script downloads the input, runs the job, uploads `out/` and `work/`, and shuts the machine down.
4. The shutdown terminates the machine, which deletes its root volume. Billing for the machine stops when it enters `shutting-down`.
5. The start command, or a person with the read profile, fetches `out/` from the bucket.

The smoke test runs on a g5.xlarge. The image build tries a g5.8xlarge, then a g5.4xlarge, the way a job tries its fallback list: `fallback` in `[cloud.image]` (Step 8 of the setup guide). Both are A10G types, because fused-ssim compiles for the builder's own GPU. They're paid runs under the same rules.

### The launch request

The runner role refuses any launch that breaks its rules, so every `run_instances` call passes these:

| Parameter | Value | Why |
|---|---|---|
| `ImageId` | Our machine image's id. For an image build, the base AMI's id. | The runner role only launches images that Amazon owns, or that our account owns and that carry the project tag. |
| `InstanceType` | The preset's type, from `config.toml` | It must be one of the 8 allowed types in Step 8 of the setup guide. |
| `MinCount`, `MaxCount` | `1` and `1` | One machine per job. |
| `IamInstanceProfile` | `{"Arn": ...}` with the stack's `JobInstanceProfileArn` output | Required. It gives the machine the job role. The ARN holds the account id, so it stays out of logs and chats. |
| `SecurityGroupIds` | `[...]` with the `SecurityGroupId` output | Required. With none, EC2 uses the VPC's default group ([RunInstances](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_RunInstances.html)), which the runner role refuses. |
| `SubnetId` | A default subnet of the VPC in the `VpcId` output, or left out | The runner role only takes a subnet in the default VPC. Left out, EC2 picks a default subnet itself ([RunInstances](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_RunInstances.html)). A machine in a non-default subnet gets no public address by default ([default subnets](https://docs.aws.amazon.com/vpc/latest/userguide/default-subnet.html)), and without one it can't reach S3 or Session Manager, because our setup makes no NAT gateway or VPC endpoints. |
| `MetadataOptions` | `{"HttpTokens": "required", "HttpEndpoint": "enabled"}` | IMDSv2 only. See "IMDSv2" below. |
| `TagSpecifications` | `project=voxelis-kiroween` on the `instance`, `volume` and `network-interface`, plus tags of our own, like the job id | The runner role refuses an instance without the project tag, and only lets tags go on in the launch itself. Tagging the volume and the network interface too makes leftovers easy to find. |
| `InstanceInitiatedShutdownBehavior` | `"terminate"` | cloud.md. IAM can't require it, and the runner role can't change it after the launch, so the request has to set it. |
| `UserData` | The script, as plain text | boto3 does the base64 encoding. See "User data" below. |
| `KeyName`, `LaunchTemplate`, `InstanceMarketOptions`, `DisableApiTermination` | Left out | The runner role refuses a key pair and a launch template, and shells go through Session Manager. Without market options the machine is On-Demand, which is what our GPU quota counts. Termination protection would stop the stop-all command from terminating the machine ([instance states](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-lifecycle.html)). |

`RunInstances` also takes a `ClientToken`. Repeating a call with the same token and the same parameters starts nothing new and returns the machine from the first call. With other parameters, the repeat fails with `IdempotentParameterMismatch`. A call that names a `SubnetId` only counts its token within that subnet's Availability Zone, so the same token with a subnet in another zone starts a second machine ([idempotency](https://docs.aws.amazon.com/ec2/latest/devguide/ec2-api-idempotency.html)). When a call passes no token, boto3 fills one in ([boto3 run_instances](https://docs.aws.amazon.com/boto3/latest/reference/services/ec2/client/run_instances.html)). Passing our own token makes a repeated call safe. If that token comes from the job id, an approved rerun of the same job needs a different one, or EC2 may return the first run's machine instead of starting a new one.

### Before the launch

- Admission is by the real GPU quota (tech.md). The start command reads the G quota (code `L-DB2E81BA`) from Service Quotas, adds up the vCPUs of our machines that aren't terminated (jobs, smoke tests and image builds, by our tag), and launches only when the chosen type's vCPUs still fit under the quota. So under our 32-vCPU quota one g5.8xlarge runs at a time, while two 16-vCPU fallback machines could run side by side; more than one of our machines may run when the quota allows, though it's still one machine per job. If no fitting type would stay under the quota, the command stops with a readable error giving the quota, the running total and the type's vCPUs, instead of letting AWS refuse the launch with `VcpuLimitExceeded`.
- Not every Availability Zone offers every type ([launch troubleshooting](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/troubleshooting-launch.html)). This lists the zones that offer one ([DescribeInstanceTypeOfferings](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeInstanceTypeOfferings.html)), and the read policy allows it:

  ```
  aws ec2 describe-instance-type-offerings --location-type availability-zone --filters Name=instance-type,Values=g5.8xlarge --query "InstanceTypeOfferings[].Location" --output text --profile kiroween-read --region us-east-1
  ```

### When the launch fails

The command reports the error and stops. It never retries by itself.

| Error | What it means |
|---|---|
| `VcpuLimitExceeded` | The account's running On-Demand machines plus this one would go over the vCPU quota ([EC2 error codes](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/errors-overview.html)). Another machine is still running, or the quota is lower than the type needs. |
| `InsufficientInstanceCapacity` | AWS has no On-Demand capacity for the type right now. AWS suggests waiting a few minutes, launching without naming a zone, or using another type ([launch troubleshooting](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/troubleshooting-launch.html)). |
| `Unsupported` | The request isn't supported, for example because the zone has constraints on that type at the moment. The message says which ([EC2 error codes](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/errors-overview.html)). |
| `UnauthorizedOperation` | The request breaks one of the runner role's rules in the table above. Its encoded message holds ARNs and the account id, so it stays out of chats. Neither of our roles has `sts:DecodeAuthorizationMessage` to decode it, so compare the request with the table instead. |

### Ending a machine by hand

The stop-all command terminates every machine with our tag that isn't terminated yet. `TerminateInstances` is safe to repeat ([idempotency](https://docs.aws.amazon.com/ec2/latest/devguide/ec2-api-idempotency.html)). It terminates rather than stops: a stopped machine still pays for its root volume ([instance states](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-lifecycle.html)), and the runner role couldn't start it again, because it has no `ec2:StartInstances`. Termination can't be undone, so check what's running first, with the first command under "Without a shell".

## Our machine image

- Our own AMI, owned by the account and tagged with the project tag. It holds ffmpeg, COLMAP built with the flags in tech.md, and the GPU packages (torch, gsplat and the depth model), which go on no other machine (tech.md).
- Private, always. It holds the GPL parts of ffmpeg and COLMAP, so it's never made public or shared (rules.md). The runner role has no action that shares, publishes, copies or exports an image or a snapshot.
- The aws-setup spec leaves the image to the spec that first starts one of AWS's Deep Learning Base AMIs, like the Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04). AWS keeps the newest one's id in a public SSM parameter, which the read policy may read ([the AMI's page](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-24-04.html)):

  ```
  aws ssm get-parameter --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id --query Parameter.Value --output text --profile kiroween-read --region us-east-1
  ```

- That AMI's 20261120 release runs Ubuntu 24.04.5 with NVIDIA driver 620.70.06 and CUDA 13.5 as the default. It supports G5, G6 and G6e machines, ships SSM Agent 3.3.4793.0, uses a gp3 root volume, and lists `/opt/dlami/nvme` as the local NVMe disk's location ([release notes](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-ami-gpubaseoss-ul2404-2026-11-21.html)).
- Every allowed type comes with one local NVMe disk, from 250 GB on the g5.xlarge to 900 GB on the g5.8xlarge ([instance specs](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)). A job can keep its work folder there, but whatever is on it is gone once the machine stops or terminates ([instance states](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-lifecycle.html)).
- The base AMI gets its row in `docs/third-party-licences.md` with the spec that first starts it (aws-setup requirements).

### Making the image

From AWS's pages:

1. Launch a builder machine (a g5.8xlarge, else a g5.4xlarge) from the base AMI, with the same launch settings as a job and a time cap in its user data that leaves room for the install. Set its root volume's size in the launch's `BlockDeviceMappings`, because CreateImage can't change volume sizes ([CreateImage API](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_CreateImage.html)) and the runner role can't resize a volume.
2. Install everything.
3. Remove what a job machine mustn't inherit. AWS says to delete the builder's user data script from `/var/lib/cloud/instances/<instance-id>/` before making an image ([user data](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html)). `cloud-init clean --logs --machine-id` does that and more: it removes cloud-init's files under `/var/lib/cloud` and its logs, and resets `/etc/machine-id`, as cloud-init advises before cloning an image ([cloud-init CLI](https://docs.cloud-init.io/en/latest/reference/cli.html)). Leave no AWS keys and no `~/.aws` files: boto3 tries the instance role last, after environment variables and those files ([boto3 credentials](https://docs.aws.amazon.com/boto3/latest/guide/credentials.html)), so they would win over the job role.
4. Stop the builder with `StopInstances`, then call `CreateImage` on the stopped machine, with the project tag on the image and on its snapshots in that same call. The runner role refuses it without both tags. CreateImage takes a running machine too, but then it reboots it by default for consistent snapshots, and `NoReboot` skips the reboot at the price of snapshots that miss whatever was still in memory ([CreateImage API](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_CreateImage.html)). AWS's pages don't say whether that reboot counts as a shutdown from inside the machine, which would terminate our builder. An API stop never triggers the terminate behaviour ([shutdown behaviour](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Using_ChangingInstanceInitiatedShutdownBehavior.html)), so stopping first avoids the question.
5. Wait for the image to become `available`, which takes several minutes and can take up to 24 hours ([creating an AMI](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/creating-an-ami-ebs.html)). Then terminate the builder.

After that:

- The image's root volume has to keep `DeleteOnTermination` true, or every job leaves a paid volume behind. It's the one setting CreateImage can change on an existing volume ([CreateImage API](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_CreateImage.html)).
- An image doesn't keep the builder's user data ([user data](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html)). A machine started from the image has a new instance id, so cloud-init treats its boot as a first boot and runs that machine's own user data ([first boot](https://docs.cloud-init.io/en/latest/explanation/first_boot.html)).
- To list our images:

  ```
  aws ec2 describe-images --owners self --filters "Name=tag:project,Values=voxelis-kiroween" --query "Images[].[ImageId,Name,CreationDate,State]" --output text --profile kiroween-read --region us-east-1
  ```

## The job bucket

The layout, from `docs/aws-setup.md` and the aws-setup design:

```
jobs/<job id>/
  input/    the video and job.json. The start command uploads them with the runner role. Kept.
  out/      the reports, job.json, job.log and the results. The job machine uploads them. Kept.
work/<job id>/
  <step>/   each step's in-between files. The job machine uploads them. Deleted 60 days after upload.
```

- It mirrors the job folder in `docs/pipeline.md`: `data/jobs/<job id>/work/` goes to `work/<job id>/`, and the rest of `data/jobs/<job id>/` goes to `jobs/<job id>/`.
- Two more folders sit under `jobs/`, so the runner role may upload to them and the job role may read and write them. `jobs/_image/<image name>/` holds the five dependency manifests (`pyproject.toml`, `uv.lock`, `.python-version`, `package.json`, `package-lock.json`) that the image-build command uploads from HEAD before the launch, because `uv.lock` alone is over the 16 KB user data limit. The builder fetches them with the AWS CLI. `jobs/_smoke/<image id>.json` is the marker a passing smoke test writes, and `jobs/_smoke/<image id>.log` holds the smoke machine's check results, pass or fail.
- Never put a finished result under `work/`, because the 60-day rule deletes everything there. S3 runs lifecycle rules once a day and rounds each date up to the next midnight UTC, so a work file can last a day or so past its 60 days, though its billing stops on the date ([AWS re:Post](https://repost.aws/knowledge-center/s3-lifecycle-rule-delay)).
- An upload that's still unfinished 7 days after it started gets cancelled, anywhere in the bucket, and its parts deleted.
- The bucket's name is the stack's `BucketName` output. The start command reads it by the stack's name, `[aws].stack_name`, and passes it to the machine in the user data.
- The bucket refuses requests without TLS, and encrypts every file with SSE-S3.

| Who | List | Read | Upload | Delete |
|---|---|---|---|---|
| The job role, on the machine | Only with a prefix under `jobs/` or `work/` | `jobs/`, `work/` | `jobs/`, `work/` | No |
| The runner role | The whole bucket | The whole bucket | `jobs/` only | No |
| The read profile | The whole bucket | The whole bucket | No | No |

- Apart from the 60-day rule, only the admin login deletes files.
- On the machine, every list names a prefix, like `work/<job id>/`. A list of the whole bucket is refused, and so is HeadBucket, which S3 checks as a list with no prefix. boto3's `upload_file` and `download_file` don't call HeadBucket, so they work (aws-setup design, "Risks and limits").
- S3 answers a download of a missing file with `404` only when the caller may list the bucket, and with `403` otherwise ([GetObject](https://docs.aws.amazon.com/AmazonS3/latest/API/API_GetObject.html)). The job role may list only with a prefix, so on the machine a missing file may show up as `403 Access Denied`. That's why `BotoAws.s3_exists` asks with a list whose prefix is the key, not with a HeadObject.

## User data

What AWS's pages say ([user data](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html), [instance metadata](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-metadata.html), [reading it](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html)):

- It's a script or cloud-init config that the machine gets at launch. On Linux, a script runs once, on the machine's first boot, as root and with no terminal to answer prompts. cloud-init runs it in its last boot stage, which holds up nothing else in the boot ([boot stages](https://docs.cloud-init.io/en/latest/explanation/boot.html)).
- A script starts with `#!` and the interpreter's path, like `#!/bin/bash`.
- It's at most 16 KB before base64. boto3 does the base64 encoding, so pass plain text.
- It isn't secret. Any software on the machine can read it from the metadata service, and `ec2:DescribeInstanceAttribute`, part of the read policy's `ec2:Describe*`, reads it from outside. So ours holds only things like the job id, the bucket name and the time cap, never keys. The machine gets AWS access from its instance role.
- cloud-init keeps a copy under `/var/lib/cloud/instances/<instance-id>/` and writes the script's output to `/var/log/cloud-init-output.log`. cloud-init's systemd units give it no time limit and send its output to the console as well ([cloud-init-main.service](https://github.com/canonical/cloud-init/blob/main/systemd/cloud-init-main.service.tmpl), [cloud-final.service](https://github.com/canonical/cloud-init/blob/main/systemd/cloud-final.service)). So a long job can run inside the script, and `get-console-output` shows what it printed.

Our own notes, not AWS's:

- The start command writes values into the script, like the job id, the bucket name and the time cap. It checks each one first, the job id against its pattern in `docs/pipeline.md`, and quotes it with `shlex.quote`.
- If the script is a file in the repo, it has to reach EC2 with LF line endings, even from Windows. A CR at the end of `#!/bin/bash` makes Linux look for an interpreter called `/bin/bash\r`, which doesn't exist. Python's text mode turns CRLF into LF when it reads a file, and a `.gitattributes` line like `*.sh text eol=lf` keeps git from adding CRs.
- The order of the script matters for money. A sketch, not tested:

  ```bash
  #!/bin/bash
  set -euo pipefail
  # The start command writes JOB_ID, BUCKET and CAP_MINUTES (from config.toml) in here.
  trap 'shutdown -h now' EXIT       # power off when the script ends, done or failed
  shutdown -h +"$CAP_MINUTES"       # the time cap, in case anything below hangs
  # Download jobs/$JOB_ID/input/, run the job, upload out/ and work/.
  ```

- The trap comes first, so even a failure on the cap's own line ends the machine. The cap comes next, so a hang anywhere later still ends it.
- The trap can upload `out/` before it powers off, so a failed job still leaves its reports and `job.log`. If that upload hangs, the cap still fires.
- If the script never runs at all, for example because of a CR on its first line, nothing ends the machine, and the stop-all command has to. An image that schedules its own cap at every boot would cover this case too.

## IMDSv2

- The metadata service at `169.254.169.254` tells the machine its instance id and region, and hands it the job role's credentials ([metadata categories](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-metadata.html)).
- With IMDSv2 required, every read needs a session token. A PUT to `/latest/api/token` with the header `X-aws-ec2-metadata-token-ttl-seconds` (1 second to 6 hours) returns one, and each GET sends it back in the header `X-aws-ec2-metadata-token`. A GET without a valid token gets `401` ([how IMDSv2 works](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/configuring-instance-metadata-service.html)).
- A token only works on the machine that asked for it. The service refuses a PUT that carries an `X-Forwarded-For` header, and the PUT's reply has a hop limit of 1 by default ([how IMDSv2 works](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/configuring-instance-metadata-service.html)). A container on the machine would need a hop limit of 2 ([considerations](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html)), so if a job ever runs in a container, the launch sets `HttpPutResponseHopLimit` to 2.
- boto3 (from 1.12.6) and the AWS CLI (from 1.16.289) use IMDSv2 by themselves ([supported SDKs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/configuring-instance-metadata-service.html)). They fetch the role's credentials, which AWS rotates, with new ones ready at least 5 minutes before the old ones expire ([role credentials](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-metadata-security-credentials.html)). So code on the machine uses a plain `boto3.Session(region_name=...)` with the region from `[aws].region`, and no profile (rules.md).
- In a shell script, call curl with `-f`. Without it, curl exits with 0 on an HTTP error and the error text ends up in the token variable ([responses](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html)):

  ```bash
  TOKEN=$(curl -fsS -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
  curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id
  ```

- The launch turns IMDSv2 on with `MetadataOptions`: `HttpTokens` set to `required`, and `HttpEndpoint` set to `enabled`, which AWS requires alongside it ([new instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/configuring-IMDS-new-instances.html)). The runner role checks `ec2:MetadataHttpTokens`, so a launch without `required` fails with `UnauthorizedOperation` ([IAM example](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ExamplePolicies_EC2.html#iam-example-instance-metadata-requireIMDSv2)).
- AWS also lets an image, or the account's defaults, require IMDSv2 ([new instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/configuring-IMDS-new-instances.html)). Neither is ours to use. An image's `imds-support` setting needs `RegisterImage` or `ModifyImageAttribute`, which the runner role doesn't have, and can't be undone. The account's IMDS defaults are an account-wide setting, and we change none of those (aws-setup design).
- Metadata requests are free but throttled per machine. AWS advises keeping the credentials until they near expiry rather than asking for them on every call ([throttling](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html)), so keep one boto3 session per process.
- Instance tags only show up in the metadata when access to them is turned on, which our jobs don't need ([metadata categories](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-metadata.html)).
- Security groups don't filter traffic to the metadata service, DNS or the time service, so our "nothing in, 443 and 80 out" group doesn't block them ([security groups](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-security-groups.html)).

## Shutting down at the end

- A shutdown from inside an EBS-backed machine stops it by default. `InstanceInitiatedShutdownBehavior=terminate` makes it terminate instead. The setting only covers shutdowns from the operating system, not `StopInstances` or the console ([shutdown behaviour](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Using_ChangingInstanceInitiatedShutdownBehavior.html)).
- `shutdown -h now` and `poweroff` start that shutdown. `halt` doesn't: it halts the CPU, and the machine keeps running and billing ([shutdown behaviour](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Using_ChangingInstanceInitiatedShutdownBehavior.html)).
- In systemd's `shutdown`, `-h` and `-P` power off and `-H` halts. `+m` schedules the shutdown m minutes ahead, `-c` cancels a scheduled one and `--show` prints it. With no time given, it waits one minute ([systemd shutdown](https://www.freedesktop.org/software/systemd/man/latest/shutdown.html)).
- The job role has no EC2 permissions, so the machine can't terminate itself through the API. The operating system's shutdown, with the terminate behaviour, is the only way it ends itself.
- Billing for the machine stops once it's `shutting-down` ([instance states](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-lifecycle.html)).
- Termination wipes the RAM and the local NVMe disk, and deletes the root volume when its `DeleteOnTermination` is true ([termination](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-ec2-instance-termination-works.html)). So the job's results, and any log needed later, like `/var/log/cloud-init-output.log`, go to the bucket before the shutdown.
- AWS doesn't promise that shutdown scripts run ([termination](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-ec2-instance-termination-works.html)), so the uploads happen in the job script, not in a shutdown hook.
- A terminated machine still shows in describe calls for a short while, and its tags fall away during that time ([instance states](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-lifecycle.html)).

## Session Manager for debugging

Session Manager gives a shell on a running job machine, with no open port and no SSH key ([Session Manager](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html)).

| It needs | What we have |
|---|---|
| SSM Agent 2.3.68.0 or later on the machine ([prerequisites](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-prerequisites.html)) | The base AMI ships 3.3.4793.0. During the image build, check that it runs with `sudo systemctl status snap.amazon-ssm-agent.amazon-ssm-agent.service` for the snap, or `sudo systemctl status amazon-ssm-agent` for the deb package ([agent status](https://docs.aws.amazon.com/systems-manager/latest/userguide/ami-preinstalled-agent.html)). |
| An instance role with `ssm:UpdateInstanceInformation` and the four `ssmmessages` channel actions ([minimal role](https://docs.aws.amazon.com/systems-manager/latest/userguide/getting-started-create-iam-instance-profile.html)) | The job role's policy in `cloud/aws-setup.yaml`. |
| HTTPS out to `ssm.<region>.amazonaws.com` and `ssmmessages.<region>.amazonaws.com`. Agents from 3.3.40.0 on use `ssmmessages` in place of `ec2messages` where it's available ([troubleshooting](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-troubleshooting.html)). | The security group's 443 rule and the machine's public address. |
| AWS CLI 1.16.12 or later and the Session Manager plugin, 1.2.764.0 or later, on the laptop ([plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)) | Each person installs the plugin. |
| `ssm:StartSession` for the person | The runner role, only on machines with our tag. |

To open a shell, in your own terminal (`docs/aws-setup.md`):

```
aws ssm start-session --target <instance-id> --profile kiroween-runner --region us-east-1
```

- Kiro doesn't open shells (`docs/aws-setup.md`). People do, in their own terminal.
- The shell runs as `ssm-user`, a user the agent makes with root permissions ([prerequisites](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-prerequisites.html)). The job's files belong to root, because user data runs as root.
- `TargetNotConnected` means the machine's agent isn't connected. A new machine needs a few minutes first ([agent status](https://docs.aws.amazon.com/systems-manager/latest/userguide/ami-preinstalled-agent.html)). A machine in another region gives the same error, so check `--region` next ([troubleshooting](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-troubleshooting.html)).
- Session Manager ends a shell after 20 minutes without input, by default ([idle timeout](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-preferences-timeout.html)). That's one of Session Manager's preferences, which everyone in the account's region shares, so we leave them as they are (aws-setup design).
- A shell doesn't keep the machine alive, and it ends when the machine does. Don't cancel the cap with `shutdown -c`. Without it, a hang keeps the machine running past the time and cost the person said yes to (rules.md).
- In a shell, `cloud-init status --wait` waits until cloud-init has finished, user data included ([boot stages](https://docs.cloud-init.io/en/latest/explanation/boot.html)), so when the job runs inside the script, it waits for the whole job. `tail -f /var/log/cloud-init-output.log` follows the script's output.

### Without a shell

The read profile is enough for these:

```
aws ec2 describe-instances --filters "Name=tag:project,Values=voxelis-kiroween" "Name=instance-state-name,Values=pending,running,shutting-down,stopping,stopped" --query "Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime,StateReason.Message]" --output text --profile kiroween-read --region us-east-1
aws ec2 get-console-output --instance-id <instance-id> --latest --query Output --output text --profile kiroween-read --region us-east-1
```

- The first lists our machines that aren't terminated, with their state and the reason for their last state change. Its `--query` leaves out the account id and the ARNs that the full output holds. Every lookup filters by our tag (rules.md).
- The second prints the boot console, which also gets cloud-init's output. `--latest` asks for the newest output, which only Nitro machines like ours support. Without it, EC2 only posts output around state changes ([console output](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/troubleshoot-unreachable-instance.html#instance-console-console-output), [GetConsoleOutput](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_GetConsoleOutput.html)). The read policy allows it on machines with our tag.
- Kiro may run the first. The console can show AWS error messages from the script, which can hold the account id. Cloud_Jobs applies the account redaction, replacing every run of exactly 12 digits with `<account>`, to the error messages, console output and job logs its commands print and to the text logs the machine uploads, so the account id stays out of chats (rules.md). So Kiro may read the redacted console output to debug a machine.

## Our rules that touch this

- cloud.md: every paid entry point calls the spend guard first. Machine types, prices and time caps come from `config.toml`. Every instance gets the project tag and `InstanceInitiatedShutdownBehavior=terminate`.
- rules.md, AWS access: everything runs in us-east-1. The runner profile is only for the start, image-build, smoke-test and stop-all commands, and the read profile is for everything else. Every aws command passes `--profile` and `--region us-east-1`, and Python on a laptop names both in `boto3.Session(profile_name=..., region_name=...)`. Code on the job machine uses its instance role. No AWS keys or account id go in the repo, chats or logs, and ARNs hold the account id. Look up only our own things, by name or by our tag.
- tech.md: admission by the real GPU quota (one machine per job, not one in the whole account), boto3 only inside `cloud/`, and the GPU packages only on the machine image.
- `docs/third-party-licences.md`: the Session Manager plugin has its row. The base AMI gets its row in the commit that first uses it.

## Open questions

- Whether CreateImage's reboot of a running machine triggers the terminate behaviour. AWS's pages don't say. The stop-first order in "Making the image" avoids it.
- How the pipeline code gets onto the machine: in the image, or fetched by the job script.
- Where in the bucket the machine's own logs go, like `cloud-init-output.log`, and how they keep out the account id that AWS error messages can hold (rules.md). The spec that writes the user data script decides.

Content from the linked AWS, cloud-init and systemd pages was rephrased for compliance with licensing restrictions.
