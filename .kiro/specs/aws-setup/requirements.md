# Requirements Document

## Introduction

The Pipeline's heavy work runs on AWS EC2, one GPU machine per job, with the job files in S3. This spec makes the AWS setup for that as one CloudFormation template, the Stack_Template (`cloud/aws-setup.yaml`), which the Account_Owner deploys once as the Stack, from CloudShell in the AWS console. The Stack makes a private Job_Bucket for the job folders, a Job_Role and a Job_Security_Group for the Job_Machines, a Runner_Role that starts and cleans up the Job_Machines, a Read_Policy for everyday use, and a Project_Group for the people who run jobs. The spec also writes the Setup_Guide (`docs/aws-setup.md`) for anyone who sets up their own copy.

The setup has to work in any AWS account, because anyone can clone the repo and run their own copy. So the Stack_Template holds no account id and none of our names, everything the Stack makes is in the region the Stack is deployed in, and our code will find the Stack's resources through the Stack_Outputs. Only the Stack's name goes in Config. Our team deploys the Stack in our team's account in us-east-1, where the G_Quota is 32 vCPUs.

Our team's account also holds an older project's resources, in other regions and in IAM and S3. The Stack, the Setup_Guide, the commits and the chats leave them exactly as they are: nothing changes, reuses, depends on, lists or names them. Everything the Stack makes gets its own name and the Project_Tag.

Out of scope:

- AWS code. The start-command spec writes the code that reads the Stack_Outputs, and adds boto3 then. This spec only defines the Stack_Outputs and documents them.
- Deploying the Stack. Only the Account_Owner deploys the Stack, never Kiro, and nothing in this spec calls AWS: cfn-lint and the Compliance_Check run locally without AWS credentials. The guard hook already blocks `cloudformation create-stack`, `update-stack`, `deploy` and `execute-change-set`.
- The IAM users, the budget alert, the Default_VPC and the G_Quota request. The Account_Owner does these by hand, following the Setup_Guide.
- The Machine_Image, and the Licence_List row for AWS's Deep Learning Base AMI. Both come with the spec that first starts the Deep Learning Base AMI.
- Tests beyond the Template_Test, and property tests. The user chose this for this spec (Requirement 17).

The user asked for 5 tasks or fewer for this spec.

### Assumptions accepted at review

The user didn't ask for these, and accepted each one at review. Each one is marked in the requirement it belongs to.

- **A1 (Requirement 13):** The budget covers the whole account. A budget for the Project_Tag alone needs the tag turned on as a cost allocation tag first, which is an account-wide billing setting.
- **A2 (Requirement 12):** IAM users can't open the Billing console until the Root_Login turns on IAM access to billing, which is an account-wide setting. So the budget step uses the Admin_Login when the Billing console lets the Admin_Login in, and the Root_Login otherwise, and the Setup_Guide leaves that setting as it is. The Paid_Plan step comes before the Admin_Login exists, so the Paid_Plan step uses the Root_Login.
- **A3 (Requirement 6):** The Runner_Role starts an instance only by launching a new one, only with the Job_Security_Group, only in a subnet of the Default_VPC and only without a key pair. Without the security group and subnet limits, a launch could skip the Job_Security_Group's rules. Without a key pair, every shell goes through Session Manager.
- **A4 (Requirements 3 and 7):** "Write" means upload. The Job_Role and the Runner_Role can't delete objects in the Job_Bucket, and the 60-day rule deletes the Work_Files. A later spec that needs deletes adds them, and the Account_Owner updates the Stack.
- **A5 (Requirement 1):** The Stack_Template takes no parameters and holds its own numbers and names (the 60 and 7 days, the ports, the Allowed_Types, the Project_Tag and the prefixes `jobs/` and `work/`), as an exception to the rule that every number and setting lives in Config, because CloudFormation can't read Config.
- **A6 (Requirement 17):** The Template_Test fails on cfn-lint's errors and warnings, not on cfn-lint's informational findings, and checks the Stack_Template for the Stack_Region. The Stack_Template turns off no cfn-lint check.
- **A7 (Requirement 12):** The Setup_Guide also says how to update the Stack when the Stack_Template changes, with a change set that the Account_Owner makes in CloudShell, reviews and then executes, and that deleting the Stack keeps the Job_Bucket, which the Account_Owner empties and deletes by hand.

### Decided at review

- **Q1 (Requirement 1, criteria 12 and 13; Requirement 12, criteria 6, 8, 10 and 11; Requirement 13, criterion 14; Requirement 15, criterion 3):** The Account_Owner deploys and updates the Stack from CloudShell, not by uploading the Stack_Template in the CloudFormation console. Uploading a template file in the console makes CloudFormation keep a copy in an S3 bucket outside the Stack. CloudFormation makes that bucket without the Project_Tag, or reuses one that the account already has for the region. `--template-body` sends the Stack_Template inside the request, so nothing outside the Stack is made or reused. `--template-body` takes at most 51,200 bytes, so the Stack_Template stays within that size. An update goes through a change set that the Account_Owner reviews and then executes, never through `update-stack`, which skips the review. CloudShell runs as the console's login and has no named profiles, so the CloudShell commands pass `--region` and no `--profile`. Kiro never runs these commands.
- **Q2 (Requirement 1, criteria 10 and 11; Requirement 2, criteria 1, 6, 7 and 8; Requirement 3, criteria 2, 3, 5 and 6; Requirement 7, criteria 4 and 7; Requirement 8, criterion 6; Requirement 12, criterion 9):** A job's Work_Files go under `work/<Job_Id>/` in the Job_Bucket, with no tag, and the job's `input/` and `out/` stay under `jobs/<Job_Id>/`. The 60-day rule's only filter is the prefix `work/`, so the rule can never reach anything under `jobs/`, and no role in the Stack gets `s3:PutObjectTagging`. This replaces the earlier design, which kept the Work_Files under `jobs/<Job_Id>/work/`, had the Job_Machines tag each Work_File for the 60-day rule, and gave the Job_Role `s3:PutObjectTagging` only on `jobs/*/work/*`. The review found two gaps in that design, and this decision closes both. First, in an IAM resource ARN, `*` also matches `/`, so `jobs/*/work/*` also matched a file in `input/` or `out/` under a folder named `work`. Second, CreateMultipartUpload's API reference doesn't say that a multipart upload's tags need `s3:PutObjectTagging`, and boto3 and the AWS CLI send files above 8 MB as multipart uploads, so a buggy upload might have tagged a file in `input/` or `out/` for the 60-day rule to delete. With no tag permission in the Stack and no tag filter in the lifecycle rules, neither gap can get a file under `jobs/` deleted. The Job_Bucket no longer mirrors `data/jobs/<Job_Id>/` exactly: the upload code, which a later spec writes, maps `data/jobs/<Job_Id>/work/` to `work/<Job_Id>/`.
- **Q3 (Requirement 13, criterion 15):** From version 2.36.50, AWS CLI v2 can end an interactive `aws configure` that writes values by offering to configure AWS skills and the AWS MCP server for an AI coding agent that it finds on the laptop, Kiro included. A yes would give Kiro AWS access outside the Read_Profile and the Runner_Profile, so the Setup_Guide tells each Operator to answer `n`.

## Glossary

- **Pipeline**: the `splatpipe/` package, which turns one phone video into a splat. The Pipeline's heavy work runs on the Job_Machines.
- **Step**: one part of the Pipeline's work on one job, like picking frames. Each Step has its own folder for its in-between files, `data/jobs/<Job_Id>/work/<name>/` on the laptop, where `<name>` is the Step's name.
- **Stack_Template**: `cloud/aws-setup.yaml`, the one CloudFormation template, in plain YAML, that makes every AWS resource this spec needs.
- **Stack**: the CloudFormation stack that the Account_Owner makes from the Stack_Template, with the name in `[aws].stack_name` in Config (`voxelis-kiroween`).
- **Stack_Region**: the region the Stack is deployed in, which `[aws].region` in Config names (`us-east-1` for our team).
- **Stack_Output**: one of the Stack's outputs. Our code reads the Stack_Outputs instead of keeping account-specific names in the repo.
- **Account_Owner**: the person who owns the AWS account that the jobs run on. Only the Account_Owner uses the AWS console.
- **Root_Login**: the AWS account's root user.
- **Admin_Login**: the IAM user with administrator access that the Account_Owner makes with the Root_Login for this setup, as a new user even when the account already has an admin IAM user, so the setup depends on no other user in the account. The Admin_Login has a console password, MFA, AWS's managed `AdministratorAccess` policy, no access keys, the Project_Tag and a name that starts with the Stack's name.
- **CloudShell**: AWS CloudShell, a shell in the browser that the Account_Owner opens from the AWS console, with the AWS CLI already in it. CloudShell runs as the console's login and has no named profiles.
- **Paid_Plan**: AWS's paid account plan. An account on AWS's free plan can't use every service, and AWS closes a free plan account when the account's credits run out or after six months.
- **Operator**: a person who runs jobs on the account, the Account_Owner included.
- **Everyday_User**: an Operator's own IAM user in the Project_Group, with access keys for the AWS CLI and no console password. The Everyday_User's keys make the Operator's Read_Profile.
- **Project_Group**: the IAM group that the Stack makes for the Operators. Adding an Everyday_User to the Project_Group is the Account_Owner's OK for that Operator to run jobs on the account.
- **Read_Policy**: the managed policy that the Stack makes for everyday use. The Read_Policy only reads, and only the Stack's own things in the Stack_Region.
- **Runner_Role**: the IAM role that the Stack makes for starting, stopping and cleaning up Job_Machines. The Project_Group's members can assume the Runner_Role without MFA.
- **Job_Role**: the IAM role that the Stack makes for the Job_Machines.
- **Job_Profile**: the instance profile that the Stack makes to hold the Job_Role.
- **Job_Machine**: an EC2 GPU instance that the Runner_Role starts, for a job, for the smoke test or for making a Machine_Image.
- **Job_Security_Group**: the security group that the Stack makes for the Job_Machines.
- **Default_VPC**: the Stack_Region's default VPC, which AWS makes in each region of a new account. The Job_Machines start in the Default_VPC's default subnets.
- **Job_Bucket**: the private S3 bucket that the Stack makes for the job folders.
- **Job_Id**: a job's id, like `20261119-142530-7f3a9c`: the UTC date as 8 digits, the UTC time as 6 digits and 6 random hex digits, joined by hyphens, so a Job_Id is always 22 characters.
- **Bucket_Job_Folder**: `jobs/<Job_Id>/` in the Job_Bucket, which holds the job's `input/` and `out/` folders, the copies of `data/jobs/<Job_Id>/input/` and `data/jobs/<Job_Id>/out/` on the laptop. The job's `work/` folder goes to the job's Bucket_Work_Folder instead.
- **Bucket_Work_Folder**: `work/<Job_Id>/` in the Job_Bucket, the copy of the job's work folder `data/jobs/<Job_Id>/work/` on the laptop, with the same Step folders inside.
- **Work_File**: an object under `work/` in the Job_Bucket.
- **Project_Tag**: the tag `project=voxelis-kiroween`, as `[aws].project_tag` in Config says.
- **Amazon_Image**: a machine image (AMI) whose owner is `amazon` (the `ec2:Owner` value), like AWS's Deep Learning Base AMIs.
- **Machine_Image**: our own AMI, which the Runner_Role makes from a Job_Machine. A Machine_Image is owned by the account and carries the Project_Tag.
- **Base_AMI_Parameter**: one of AWS's public SSM parameters under `/aws/service/deeplearning/ami/x86_64/` whose name starts with `base-`, each holding the id of the latest Deep Learning Base AMI of one kind, like `/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id`.
- **Allowed_Type**: one of the 8 instance types that the Runner_Role may start: g5.xlarge, g5.2xlarge, g5.4xlarge, g5.8xlarge, g6.2xlarge, g6.4xlarge, g6e.2xlarge and g6e.4xlarge.
- **G_Quota**: the Service Quotas quota "Running On-Demand G and VT instances" (quota code `L-DB2E81BA`) in the Stack_Region, counted in vCPUs.
- **IMDSv2**: version 2 of the EC2 instance metadata service, which only answers requests that carry a session token.
- **Debug_Shell**: a shell on a Job_Machine through AWS Systems Manager Session Manager, which needs no open port.
- **Session_Manager_Plugin**: AWS's Session Manager plugin for the AWS CLI, which an Operator needs on the laptop to open a Debug_Shell.
- **Read_Profile**: the AWS CLI profile that `[aws].read_profile` in Config names (`kiroween-read`), made from an Everyday_User's access keys. Kiro uses the Read_Profile.
- **Runner_Profile**: the AWS CLI profile that `[aws].runner_profile` in Config names (`kiroween-runner`), which assumes the Runner_Role with the Read_Profile's keys. Kiro uses the Runner_Profile only inside our start, smoke-test and stop-all commands.
- **Approval_Prompt**: Kiro's prompt that asks the person in the chat to allow a command before Kiro runs the command.
- **Spend_Guard**: the function in `cloud/spend.py` that every paid entry point calls first. The Spend_Guard refuses the run unless `APPROVED_SPEND=1` is set on that one command.
- **Other_Resource**: anything in the AWS account that this setup doesn't make, by the Stack or by hand following the Setup_Guide: resources, IAM users, roles, policies, buckets and stacks, the older project's included. The Default_VPC is not an Other_Resource here, because the user chose to start the Job_Machines in the Default_VPC.
- **Compliance_Check**: the `check_cloudformation_template_compliance` tool of the aws-infrastructure-as-code Kiro power, which checks a template against cfn-guard's AWS security rules, locally.
- **Unfit_Rule**: a Compliance_Check rule that doesn't fit a small private setup: a rule that asks for access logs, versioning, replication, object lock or KMS keys; a rule that fails although the Stack already does what the rule wants, like the public read and write ACL rule on a bucket with ACLs off and all public access blocked; the wildcard-resource rule on a statement whose actions only take `*` as the resource, like EC2's describe actions; and the open-outbound rule for the Job_Security_Group's HTTPS and HTTP outbound rules.
- **Suppression**: a rule id in a resource's `Metadata: guard: SuppressedRules` list in the Stack_Template, which tells the Compliance_Check to skip that rule for that resource.
- **Template_Test**: the test in `tests/` that runs cfn-lint on the Stack_Template.
- **Allowed_Licences**: the permissive licences that the Rules_Steering lists for our code and our Python and npm dependencies.
- **Builder**: whoever carries out this spec's tasks: Kiro, or a person on the team.
- **Config**: `config.toml` at the repo root, read with `tomllib`.
- **Setup_Guide**: `docs/aws-setup.md`.
- **Licence_List**: `docs/third-party-licences.md`.
- **Rules_Steering**: `.kiro/steering/rules.md`.
- **Tech_Steering**: `.kiro/steering/tech.md`.
- **Structure_Steering**: `.kiro/steering/structure.md`.
- **README**: `README.md` at the repo root.
- **Project_File**: `pyproject.toml` at the repo root, with `uv.lock` next to `pyproject.toml`.
- **Test_Suite**: the tests in `tests/`, run with `uv run pytest`.

## Requirements

### Requirement 1: One plain template

**User Story:** As the Account_Owner, I want every AWS resource of the Pipeline in one plain CloudFormation template that I deploy once, so that any account can run its own copy without a CDK bootstrap stack and without anything account-specific in the repo.

**Assumption accepted at review:** A5 (criteria 2 and 11). **Decided at review:** Q1 (criteria 12 and 13) and Q2 (criteria 10 and 11).

#### Acceptance Criteria

1. THE Stack_Template SHALL be one CloudFormation template in plain YAML at `cloud/aws-setup.yaml`, with no CDK, no transform, no macro and no nested stack, so the account needs no CDK bootstrap stack and no other stack.
2. THE Stack_Template SHALL take no parameters, so the Account_Owner gives only the Stack's name and the Project_Tag when deploying the Stack.
3. THE Stack_Template SHALL take the account id, the region and the partition only from CloudFormation's pseudo parameters (`AWS::AccountId`, `AWS::Region` and `AWS::Partition`), so the Stack_Template holds no account id and no region name.
4. THE Stack_Template SHALL leave the Job_Bucket's name and the Job_Security_Group's name to CloudFormation.
5. THE Stack_Template SHALL leave each IAM resource's name to CloudFormation, or build the name from the Stack's name and the Stack_Region, because IAM names are account-wide and a copy of the Stack can run in another region of the same account.
6. THE Stack_Template SHALL use the account id in no name.
7. THE Stack_Template SHALL put the Project_Tag on each resource whose CloudFormation type takes tags: the Job_Bucket, the Job_Role, the Runner_Role and the Job_Security_Group.
8. THE Stack_Template SHALL refer only to the resources the Stack makes, the Stack itself, the account's root principal, the EC2 service, the Default_VPC and its subnets (through the Job_Security_Group), the Job_Machines and their volumes and network interfaces, Amazon_Images, Machine_Images and their snapshots, the Base_AMI_Parameters, the G_Quota, the Debug_Shell sessions and Session Manager's own shell document, and to no Other_Resource. The only exceptions are the actions whose only resource is `*` (EC2's describe actions and three of the Session Manager agent's channel actions), which the Stack_Template limits to the Stack_Region.
9. THE Stack_Template SHALL name no person and no Other_Resource, in the Stack_Template's values or in the Stack_Template's comments.
10. THE Stack_Template SHALL hold the Project_Tag with the key and the value of `[aws].project_tag` in Config, and the Job_Bucket's two prefixes as `jobs/` for the Bucket_Job_Folders and `work/` for the Bucket_Work_Folders.
11. THE Stack_Template SHALL hold the Stack_Template's own numbers and names (the 60 and 7 days, the ports, the Allowed_Types, the Project_Tag and the prefixes `jobs/` and `work/`), as an exception to the rule that every number and setting lives in Config, because CloudFormation can't read Config.
12. THE Stack_Template SHALL be at most 51,200 bytes, the most that `--template-body` takes, so the Account_Owner can deploy and update the Stack from CloudShell.
13. WHEN the Builder has written or changed the Stack_Template, THE Builder SHALL check the Stack_Template's size against criterion 12 before committing the Stack_Template, as a check by hand and not as a test, because Requirement 17 keeps the new tests to the Template_Test.

### Requirement 2: The job bucket

**User Story:** As an Operator, I want one private, encrypted bucket for the job folders that deletes the in-between files by itself, so that the jobs' files stay private and the Work_Files don't pile up.

**Decided at review:** Q2 (criteria 1, 6, 7 and 8).

#### Acceptance Criteria

1. THE Stack_Template SHALL make one S3 bucket, the Job_Bucket, which holds one Bucket_Job_Folder and one Bucket_Work_Folder per job.
2. THE Job_Bucket SHALL block all public access, with all four Block Public Access settings on.
3. THE Job_Bucket SHALL have ACLs turned off, with object ownership set to bucket owner enforced.
4. THE Job_Bucket SHALL encrypt each new object at rest by default with S3-managed keys (SSE-S3).
5. THE Job_Bucket's bucket policy SHALL refuse every request to the Job_Bucket that doesn't use TLS.
6. WHEN an object under `work/` in the Job_Bucket is 60 days old, THE Job_Bucket SHALL delete the object, through a lifecycle rule whose filter is the prefix `work/`, with no tag filter.
7. WHEN a multipart upload to the Job_Bucket is still unfinished 7 days after the upload started, THE Job_Bucket SHALL abort the upload, through a second lifecycle rule with no filter, separate from the 60-day rule, so the abort rule covers the whole Job_Bucket, `jobs/` included, for a broken upload of a large video.
8. THE Job_Bucket's lifecycle rules SHALL delete only objects under `work/`, so the files under `jobs/` stay until someone deletes those files.
9. THE Stack_Template SHALL set the Job_Bucket's `DeletionPolicy` to `RetainExceptOnCreate` and the Job_Bucket's `UpdateReplacePolicy` to `Retain` (`UpdateReplacePolicy` has no `RetainExceptOnCreate` value), so deleting the Stack, or a Stack update that replaces the Job_Bucket, keeps the Job_Bucket and the files in the Job_Bucket, while a failed first create of the Stack rolls back and deletes the still-empty Job_Bucket, which leaves nothing behind.

### Requirement 3: The job role

**User Story:** As the Account_Owner, I want the Job_Machines to reach only the Job_Bucket and Session Manager, so that a Job_Machine can't touch anything else in the account.

**Assumption accepted at review:** A4 (criteria 3 and 5). **Decided at review:** Q2 (criteria 2, 3, 5 and 6).

#### Acceptance Criteria

1. THE Stack_Template SHALL make the Job_Role, which only the EC2 service can assume, and the Job_Profile, which holds the Job_Role.
2. THE Job_Role SHALL allow listing the Job_Bucket under `jobs/` and `work/`, and reading the Job_Bucket's objects under `jobs/` and `work/`.
3. THE Job_Role SHALL allow uploading objects under `jobs/` and `work/` in the Job_Bucket, with no permission to delete objects.
4. THE Job_Role SHALL allow only the actions that the Session Manager agent on a Job_Machine needs for a Debug_Shell, and only in the Stack_Region.
5. THE Job_Role SHALL hold only the permissions in criteria 2 to 4, so the Job_Role reaches no other bucket, deletes no object and reaches nothing else in the account.
6. THE Job_Role SHALL hold no permission to tag S3 objects (`s3:PutObjectTagging`), in the Job_Bucket or anywhere else, because the 60-day rule finds the Work_Files by the prefix `work/` alone, so a Job_Machine uploads each Work_File with no tag.

Note: criterion 4 rules out AWS's `AmazonSSMManagedInstanceCore` policy, which also lets a machine read every SSM parameter in the account, in every region.

### Requirement 4: The job security group

**User Story:** As the Account_Owner, I want the Job_Machines in a security group that lets nothing in and only HTTPS and HTTP out, so that no port is open and a Job_Machine can still reach AWS and Ubuntu's package mirrors.

#### Acceptance Criteria

1. THE Stack_Template SHALL make the Job_Security_Group with no VPC id, so the Job_Security_Group lands in the Stack_Region's Default_VPC.
2. THE Stack_Template SHALL make no VPC, subnet, route table or gateway.
3. THE Job_Security_Group SHALL have no inbound rules.
4. THE Job_Security_Group SHALL allow outbound traffic only on TCP port 443 and TCP port 80, to any IPv4 address (`0.0.0.0/0`), with port 80 there for Ubuntu's apt mirrors.

Note: security groups don't filter traffic to the Amazon DNS server, the instance metadata service or the Amazon Time Sync Service, so a Job_Machine still looks up names, gets the Job_Role's credentials and keeps the Job_Machine's clock right.

### Requirement 5: The runner role

**User Story:** As an Operator, I want one role I can assume without MFA, which starts and cleans up our machines and also reads what the Read_Policy reads, so that Kiro can run our start command after I click allow, with only the Runner_Profile.

#### Acceptance Criteria

1. THE Stack_Template SHALL make the Runner_Role with a trust policy whose principal is the account's root principal, so that only principals in the account whose own policies allow `sts:AssumeRole` on the Runner_Role can assume the Runner_Role, because an IAM group can't be a principal in a trust policy.
2. THE Runner_Role's trust policy SHALL ask for no MFA, because Kiro runs the start command after the Approval_Prompt and can't type an MFA code.
3. THE Stack_Template SHALL attach the Read_Policy to the Runner_Role, so the start command needs only the Runner_Profile.
4. THE Runner_Role SHALL hold only the Read_Policy and the permissions in Requirements 6 and 7.
5. THE Runner_Role SHALL keep each Machine_Image and each snapshot private to the account, with no permission to share, publish, copy or export a Machine_Image or a snapshot, because a Machine_Image holds GPL command line tools that we don't distribute.

### Requirement 6: Starting job machines

**User Story:** As the Account_Owner, I want the Runner_Role to start only our kind of machine, so that a mistake or a bad command can't start any other machine in the account.

**Assumption accepted at review:** A3 (criteria 1 and 7).

#### Acceptance Criteria

1. THE Runner_Role SHALL allow starting an instance only by launching a new instance (`RunInstances`), and only in the Stack_Region.
2. THE Runner_Role SHALL allow launching an instance only of an Allowed_Type.
3. THE Runner_Role SHALL allow launching an instance only from an Amazon_Image, or from an AMI that the account owns and that carries the Project_Tag.
4. THE Runner_Role SHALL allow launching an instance only when the launch request gives the new instance the Project_Tag.
5. THE Runner_Role SHALL allow launching an instance only with the Job_Profile, and allow passing only the Job_Role, and only to the EC2 service.
6. THE Runner_Role SHALL allow launching an instance only when the launch request sets the new instance to require IMDSv2.
7. THE Runner_Role SHALL allow launching an instance only with the Job_Security_Group, only in a subnet of the Default_VPC that holds the Job_Security_Group, and only without a key pair.
8. THE Runner_Role SHALL allow adding tags only in the request that creates the instance, the Machine_Image or the snapshot that gets the tags, so the Runner_Role can't put the Project_Tag on anything that already exists.

Note: criterion 3 pins no AMI id. The Amazon_Images include AWS's Deep Learning Base AMIs.

### Requirement 7: Stopping machines, images, uploads and shells

**User Story:** As an Operator, I want the Runner_Role to stop and clean up only our own machines and images, upload job files and open a Debug_Shell, so that we can run and fix jobs without reaching anything else in the account.

**Assumption accepted at review:** A4 (criterion 4). **Decided at review:** Q2 (criteria 4 and 7).

#### Acceptance Criteria

1. THE Runner_Role SHALL allow stopping and terminating only instances in the Stack_Region that carry the Project_Tag.
2. THE Runner_Role SHALL allow making a Machine_Image only from an instance in the Stack_Region that carries the Project_Tag, and only when the request gives the new Machine_Image and the Machine_Image's snapshots the Project_Tag.
3. THE Runner_Role SHALL allow deregistering only AMIs, and deleting only snapshots, that are in the Stack_Region and carry the Project_Tag.
4. THE Runner_Role SHALL allow uploading objects only under `jobs/` in the Job_Bucket, for a job's video in `input/`, with no permission to delete objects.
5. THE Runner_Role SHALL allow opening a Debug_Shell only on instances in the Stack_Region that carry the Project_Tag.
6. THE Runner_Role SHALL allow ending or resuming only the Debug_Shell sessions that the Runner_Role started.
7. THE Stack_Template SHALL give no role in the Stack the permission to tag S3 objects (`s3:PutObjectTagging`), in the Job_Bucket or anywhere else, because the Job_Bucket's lifecycle rules have no tag filter, so no upload with the Runner_Role or the Job_Role needs a tag.

### Requirement 8: The read policy

**User Story:** As an Operator, I want a read-only policy for everyday use that reaches only our own things in the Stack_Region, so that Kiro can check on jobs with the Read_Profile and can't start, change or delete anything with the Read_Profile.

**Decided at review:** Q2 (criterion 6).

#### Acceptance Criteria

1. THE Stack_Template SHALL make the Read_Policy as a managed policy.
2. THE Read_Policy SHALL allow EC2's describe actions only in the Stack_Region.
3. THE Read_Policy SHALL allow reading the console output only of instances in the Stack_Region that carry the Project_Tag.
4. THE Read_Policy SHALL allow reading the G_Quota and the requests to raise the G_Quota, only in the Stack_Region.
5. THE Read_Policy SHALL allow reading the Base_AMI_Parameters, only in the Stack_Region.
6. THE Read_Policy SHALL allow listing the whole Job_Bucket and reading each object in the Job_Bucket, under `jobs/` and `work/` alike, with no listing of the account's buckets.
7. THE Read_Policy SHALL allow reading the Stack's description, outputs, resources and events, and no other stack.
8. THE Read_Policy SHALL hold only the read permissions in criteria 2 to 7, so the Read_Policy starts, changes and deletes nothing, reads nothing in IAM and reads nothing outside the Stack_Region.

### Requirement 9: The project group

**User Story:** As the Account_Owner, I want one IAM group that gives the group's members the Read_Policy and the right to assume the Runner_Role, so that adding an Everyday_User to the group is all I do to approve an Operator.

#### Acceptance Criteria

1. THE Stack_Template SHALL make the Project_Group with the Read_Policy attached.
2. THE Stack_Template SHALL attach to the Project_Group a managed policy that allows `sts:AssumeRole` on the Runner_Role and on nothing else.
3. THE Project_Group SHALL hold only the policies in criteria 1 and 2, with no self-service IAM permissions.
4. THE Stack_Template SHALL make no IAM user and add no member to the Project_Group, so the Account_Owner adds each Everyday_User by hand.

### Requirement 10: Stack outputs

**User Story:** As the team member who writes the start command, I want the Stack_Outputs to give every id and ARN our code needs, so that our code finds the Stack's resources by the Stack's name and the repo holds nothing account-specific.

#### Acceptance Criteria

1. THE Stack_Template SHALL give the Stack these Stack_Outputs: `BucketName` (the Job_Bucket's name), `JobRoleArn` (the Job_Role's ARN), `JobInstanceProfileArn` (the Job_Profile's ARN), `RunnerRoleArn` (the Runner_Role's ARN), `SecurityGroupId` (the Job_Security_Group's id), `VpcId` (the id of the Default_VPC that holds the Job_Security_Group) and `GroupName` (the Project_Group's name).
2. THE Stack_Template SHALL give each Stack_Output a description of what the Stack_Output holds.
3. THE Stack_Template SHALL give no Stack_Output an export name, so the Stack adds no account-wide export names.
4. THE Config SHALL hold the Stack's name as `stack_name = "voxelis-kiroween"` in `[aws]`, with the other `[aws]` keys unchanged.
5. THE Setup_Guide SHALL list each Stack_Output with what the Stack_Output holds and who uses the Stack_Output.

Note for the start-command spec: the ARN outputs hold the account id, so our code keeps the ARNs out of logs and out of what Kiro prints in the chat.

### Requirement 11: The compliance check

**User Story:** As the Account_Owner, I want the template checked against AWS's security rules, with the real problems fixed and the rules that don't fit a small private setup suppressed with a reason, so that the setup is safe without things we'd pay for and never use.

#### Acceptance Criteria

1. WHEN the Builder has written or changed the Stack_Template, THE Builder SHALL run the Compliance_Check on the Stack_Template before committing the Stack_Template.
2. THE Stack_Template SHALL pass the Compliance_Check, with each failing rule fixed or suppressed.
3. THE Stack_Template SHALL fix the rules that apply to the Stack, including TLS-only requests to the Job_Bucket (`S3_BUCKET_SSL_REQUESTS_ONLY`), managed policies instead of inline policies (`IAM_NO_INLINE_POLICY_CHECK`), explicit outbound rules (`SECURITY_GROUP_MISSING_EGRESS_RULE`) and no all-protocol outbound rule (`SECURITY_GROUP_EGRESS_ALL_PROTOCOLS_RULE`).
4. THE Stack_Template SHALL suppress only Unfit_Rules, each with a Suppression in the Metadata of the resource that the Unfit_Rule fails on.
5. THE Stack_Template SHALL give each Suppression a one-line reason, as a YAML comment next to the rule id.
6. THE Stack_Template SHALL suppress an Unfit_Rule instead of adding what the Unfit_Rule asks for, so the Stack makes no access logs, versioning, replication, object lock or KMS keys.
7. IF the Compliance_Check fails a rule that is not an Unfit_Rule and that the Builder can't fix within this spec's requirements, THEN THE Builder SHALL ask the user before adding a Suppression for the rule.

Note: running the Compliance_Check on drafts showed these Unfit_Rules: `S3_BUCKET_LOGGING_ENABLED`, `S3_BUCKET_VERSIONING_ENABLED`, `S3_BUCKET_REPLICATION_ENABLED`, `S3_BUCKET_DEFAULT_LOCK_ENABLED`, `S3_BUCKET_NO_PUBLIC_RW_ACL` (which fails even with ACLs off and all public access blocked, while `AccessControl: Private` would make cfn-lint warn with W3045), `IAM_POLICYDOCUMENT_NO_WILDCARD_RESOURCE` on statements whose actions only take `*`, and `EC2_SECURITY_GROUP_EGRESS_OPEN_TO_WORLD_RULE`, for the outbound rules the user agreed to. The final list comes from running the Compliance_Check on the real Stack_Template.

### Requirement 12: Setup guide: the account and the stack

**User Story:** As someone running their own copy, I want a guide that takes me from an AWS account to a deployed Stack, so that I can set up AWS without knowing our team or our account.

**Assumptions accepted at review:** A2 (criterion 3) and A7 (criteria 11 and 12). **Decided at review:** Q1 (criteria 6, 8, 10 and 11) and Q2 (criterion 9).

#### Acceptance Criteria

1. THE Setup_Guide SHALL give the Account_Owner's steps in this order: the Paid_Plan, the Admin_Login, the Default_VPC, deploying the Stack, the Everyday_Users and the Project_Group, the budget alert, each Operator's AWS CLI profiles, and the G_Quota.
2. THE Setup_Guide SHALL start with moving the account to the Paid_Plan in the Billing console with the Root_Login, and say that an account on AWS's free plan can't use every service and closes when the account's credits run out.
3. IF the Billing console refuses the Admin_Login at the budget step, THEN THE Setup_Guide SHALL have the Account_Owner make the budget with the Root_Login, and leave the account's setting for IAM access to billing as the setting is.
4. THE Setup_Guide SHALL have the Account_Owner make a new Admin_Login with the Root_Login right after the Paid_Plan, even when the account already has an IAM user with administrator access, so the setup depends on no other user in the account, and use the Admin_Login instead of the Root_Login from then on, except for the budget in criterion 3.
5. THE Setup_Guide SHALL have the Account_Owner check that the Stack_Region has a Default_VPC, and make a Default_VPC in the VPC console (Actions, Create default VPC) when the Default_VPC was deleted.
6. THE Setup_Guide SHALL have the Account_Owner deploy the Stack once from CloudShell, opened from the console with the Admin_Login: remove any older copy with `rm -f aws-setup.yaml`, upload `cloud/aws-setup.yaml` to CloudShell with CloudShell's Actions, Upload file, then run `aws cloudformation create-stack` with `--stack-name` and the name in `[aws].stack_name`, `--template-body file://aws-setup.yaml`, `--capabilities CAPABILITY_IAM`, `--tags Key=project,Value=voxelis-kiroween` and `--region <your-region>`, where `<your-region>` is the value of `[aws].region`, and wait for the Stack's create to finish.
7. THE Setup_Guide SHALL say that the Project_Tag also goes on the Stack as a stack tag because the Project_Group, the Job_Profile, the managed policies and the bucket policy take no tags in CloudFormation.
8. THE Setup_Guide SHALL say that only the Account_Owner deploys or updates the Stack, and that Kiro never runs the commands that deploy or update the Stack.
9. THE Setup_Guide SHALL describe what the Stack makes and the Job_Bucket's layout: the copies of a job's `input/` and `out/` go under `jobs/<Job_Id>/input/` and `jobs/<Job_Id>/out/` and stay until someone deletes those files, the copy of the job's `work/` goes under `work/<Job_Id>/` and the Job_Bucket deletes each Work_File 60 days after the Work_File's upload, and the Job_Bucket aborts each multipart upload that is still unfinished 7 days after the upload started.
10. THE Setup_Guide SHALL say, in words that mention no other project, not to upload the Stack_Template in the CloudFormation console, because the console has CloudFormation keep a copy of an uploaded template in an S3 bucket outside the Stack, which CloudFormation makes without the Project_Tag or reuses when the account already has one for the Stack_Region, while `--template-body` sends the Stack_Template inside the request, so nothing outside the Stack is made or reused.
11. THE Setup_Guide SHALL say how the Account_Owner updates the Stack when the Stack_Template changes: remove the old copy with `rm -f aws-setup.yaml` and upload the new Stack_Template to CloudShell, make a change set with `aws cloudformation create-change-set`, `--template-body file://aws-setup.yaml`, `--capabilities CAPABILITY_IAM`, `--tags Key=project,Value=voxelis-kiroween` and `--region <your-region>`, with the same stack tag as the create so the Stack keeps the Project_Tag, review the change set, then execute the change set, and never use `aws cloudformation update-stack`, which skips the review.
12. THE Setup_Guide SHALL say how the Account_Owner removes the setup, in this order: take every Everyday_User out of the Project_Group, delete the Stack, then empty and delete the Job_Bucket by hand, because deleting the Stack keeps the Job_Bucket and the Job_Bucket's files.
13. THE Setup_Guide SHALL say which Config values a copy changes to deploy in another region or under another name: `[aws].region` and `[aws].stack_name`.
14. THE Setup_Guide SHALL give the Admin_Login a console password, MFA, AWS's managed `AdministratorAccess` policy, the Project_Tag and a name that starts with the Stack's name.

### Requirement 13: Setup guide: people, budget and profiles

**User Story:** As the Account_Owner, I want the guide to say how to give each Operator their own keys, set a budget alert and set up the two profiles, so that nobody works as root, Kiro never holds admin keys, and every Operator's profiles work the same way.

**Assumption accepted at review:** A1 (criterion 6). **Decided at review:** Q1 (criterion 14) and Q3 (criterion 15).

#### Acceptance Criteria

1. THE Setup_Guide SHALL have the Account_Owner make one Everyday_User per Operator, the Account_Owner included, with access keys for the AWS CLI, no console password, the Project_Tag and a name that starts with the Stack's name.
2. THE Setup_Guide SHALL have the Account_Owner add each Everyday_User to the Project_Group, and say that adding an Everyday_User to the Project_Group is the Account_Owner's OK for that Operator to run jobs on the account.
3. THE Setup_Guide SHALL have the Account_Owner hand each Operator the Operator's access keys privately, and say that access keys go only into the Operator's own AWS CLI files, and so not into the repo, a chat or a log.
4. THE Setup_Guide SHALL give the Admin_Login no access keys, and say that the Read_Profile and the Runner_Profile use no admin user's keys, so Kiro never gets admin keys.
5. THE Setup_Guide SHALL say that only the Account_Owner uses the AWS console, with the Admin_Login and MFA, and that every other Operator uses only the AWS CLI.
6. THE Setup_Guide SHALL have the Account_Owner make a monthly cost budget for an amount that the Account_Owner picks, with an email alert and the Project_Tag, and say that the alert only sends an email and stops nothing.
7. THE Setup_Guide SHALL have each Operator install the AWS CLI v2 and set up the Read_Profile from the Operator's Everyday_User's access keys, in the Stack_Region.
8. THE Setup_Guide SHALL have each Operator set up the Runner_Profile to assume the Runner_Role named by the `RunnerRoleArn` Stack_Output, with the Read_Profile as the source profile, in the Stack_Region.
9. THE Setup_Guide SHALL have each Operator set up the profiles in the Operator's own terminal and not through Kiro, and put the Runner_Role's ARN only into the Operator's own AWS CLI config file, because the ARN holds the account id.
10. THE Setup_Guide SHALL give a check for each profile whose output holds no account id, no key and no ARN, like reading the Stack's status, and say that AWS error messages can hold the account id.
11. THE Setup_Guide SHALL say to name the profile in each AWS CLI command run on a laptop with `--profile` and not with `AWS_PROFILE`, because AWS keys in environment variables beat `AWS_PROFILE`, and to name the region in each AWS CLI command with `--region`, because an `AWS_REGION` or `AWS_DEFAULT_REGION` environment variable beats the profile's region, even with `--profile`.
12. THE Setup_Guide SHALL say that the Read_Profile's keys can also assume the Runner_Role, because the Project_Group allows that, so the line between the two profiles is Kiro's Approval_Prompt, the Spend_Guard and our rules, and not IAM.
13. THE Setup_Guide SHALL say that an Operator opens a Debug_Shell with the Runner_Profile and the Session_Manager_Plugin.
14. THE Setup_Guide SHALL say that the commands run in CloudShell pass `--region` and no `--profile`, because CloudShell runs as the console's login and has no named profiles, and that these commands are the one exception to naming the profile with `--profile`.
15. THE Setup_Guide SHALL tell each Operator to answer `n` when the interactive `aws configure` offers to configure AWS skills and the AWS MCP server for an AI coding agent, because Kiro gets AWS access only through the Read_Profile and the Runner_Profile.

### Requirement 14: Setup guide: the GPU quota

**User Story:** As someone running their own copy, I want the guide to say which GPU quota the jobs need and how to ask for more, so that I know why no job starts in a new account and what to ask AWS for.

#### Acceptance Criteria

1. THE Setup_Guide SHALL name the G_Quota as "Running On-Demand G and VT instances" with the quota code `L-DB2E81BA`, and say that the G_Quota counts vCPUs per account and per region.
2. THE Setup_Guide SHALL give the vCPUs that each instance size needs: 4 for an xlarge, 8 for a 2xlarge, 16 for a 4xlarge and 32 for an 8xlarge.
3. THE Setup_Guide SHALL list each Allowed_Type with what the Allowed_Type is for: g5.xlarge for the smoke test, g5.8xlarge for fast and default jobs (g5.4xlarge where the G_Quota is under 32 vCPUs), g5.4xlarge for making the Machine_Image, g6e.4xlarge for quality jobs, and g6.4xlarge, g5.2xlarge, g6.2xlarge and g6e.2xlarge as fallbacks.
4. THE Setup_Guide SHALL say that a new AWS account starts with a G_Quota of 0, and that no job can start until AWS approves an increase.
5. THE Setup_Guide SHALL say how to ask for an increase of the G_Quota in the Service Quotas console, in the Stack_Region, and to ask for 32 vCPUs so fast and default jobs can run on a g5.8xlarge.
6. THE Setup_Guide SHALL give a command that reads the G_Quota with the Read_Profile.
7. THE Setup_Guide SHALL say that the Runner_Role refuses each instance type outside the Allowed_Types, so using another type needs a change to the Stack_Template and an update of the Stack.

### Requirement 15: What the guide names and suggests

**User Story:** As the Account_Owner of an account that holds Other_Resources, I want the guide to name nothing of ours and to change nothing else in the account, so that following the guide leaves every Other_Resource exactly as before.

**Decided at review:** Q1 (criterion 3).

#### Acceptance Criteria

1. THE Setup_Guide SHALL name no person, no team member, no account id and no region, and use placeholders and the values in Config instead.
2. THE Setup_Guide SHALL name, list and describe no Other_Resource.
3. WHERE the Setup_Guide gives an AWS CLI command, THE Setup_Guide SHALL use a command that names the region with `--region <your-region>`, names the profile with `--profile` when the command runs on a laptop, and looks up only the Stack, the Stack's change sets, the Stack's resources, the G_Quota or the Base_AMI_Parameters, by name or by the Project_Tag. The commands run in CloudShell name no profile, and the command that opens a Debug_Shell on one Job_Machine by its instance id looks nothing up.
4. THE Setup_Guide SHALL suggest no account-wide setting that changes how Other_Resources or other users work, such as an account password policy, account-level S3 Block Public Access, EC2's Allowed AMIs setting or IAM access to billing.

### Requirement 16: Config, steering, README and licences

**User Story:** As a team member, I want the steering, the README and the Licence_List to match the new setup, so that Kiro and the team plan with the right quota, name the region in every AWS call, find the guide, and know each new tool's licence.

#### Acceptance Criteria

1. THE Tech_Steering's quota line SHALL say that our G_Quota allows 32 vCPUs of G instances at a time in us-east-1, that fast and default jobs run on a g5.8xlarge with 32 vCPUs on 16 cores and one NVIDIA A10G, that one job machine runs at a time, and that a new AWS account starts at 0 and has to ask AWS for more before any job can start.
2. THE Structure_Steering's one line that starts with ``- `docs/`:`` SHALL name `aws-setup.md` next to `pipeline.md`, so `tests/test_docs.py` still finds `pipeline.md` on that line.
3. THE Structure_Steering's `cloud/` line SHALL say that the Stack_Template lives in `cloud/`.
4. THE Rules_Steering SHALL list CNRI-Python, the permissive licence of Python's early code, among the Allowed_Licences.
5. THE README SHALL link to the Setup_Guide.
6. THE Project_File SHALL list cfn-lint in the dev dependency group only.
7. THE Licence_List SHALL hold a cfn-lint row with cfn-lint's licence (MIT-0), each package that adding cfn-lint puts in `uv.lock` with the package's licence, and how we use cfn-lint: the Template_Test runs cfn-lint on the Stack_Template, in the dev group only.
8. THE Licence_List SHALL hold a row for the MCP server that runs the Compliance_Check (`awslabs.aws-iac-mcp-server`), with the server's licence and how we use the server: Kiro runs the Compliance_Check through the aws-infrastructure-as-code power, and our code doesn't import or ship the server.
9. THE Licence_List SHALL hold a row for the Session_Manager_Plugin, with the Session_Manager_Plugin's licence (Apache-2.0), as a separate command line tool that Operators run to open a Debug_Shell, not vendored, linked or shipped, and Installed "Not yet".
10. WHEN the Builder adds cfn-lint, THE Builder SHALL check the licence of each package that the addition puts in `uv.lock` before committing.
11. IF a package that adding cfn-lint puts in `uv.lock` has a licence outside the Allowed_Licences, THEN THE Builder SHALL stop and ask the user before committing.
12. THE Builder SHALL add each Licence_List row in the same commit as the dependency or tool that the row covers.
13. THE Rules_Steering's `## AWS access` section SHALL hold the two lines in the second note below, right after the line that starts "In Python on our laptops, always name the profile with `boto3.Session(profile_name=...)`", with every other line of that section unchanged.

Note: cfn-lint 1.61.0 needs PyYAML (MIT), jsonpatch and jsonpointer (BSD-3-Clause), networkx (BSD-3-Clause), sympy and mpmath (BSD-3-Clause), regex (Apache-2.0 and CNRI-Python) and typing_extensions (PSF-2.0). The user OKed CNRI-Python for regex. `awslabs.aws-iac-mcp-server` 1.0.27 is Apache-2.0 on PyPI.

Note: the two lines for criterion 13, word for word:

```markdown
- aws commands also pass `--region us-east-1`. An `AWS_REGION` or `AWS_DEFAULT_REGION` environment variable beats the profile's region, even with `--profile`.
- Python also passes `region_name` from `[aws].region` in `config.toml` to `boto3.Session`.
```

### Requirement 17: Tests and building safely

**User Story:** As a team member, I want the template checked by a quick offline test and the spec built without touching AWS, so that the tests are free and safe and nothing changes in the account before the Account_Owner deploys the Stack.

**Assumption accepted at review:** A6 (criteria 1 and 3).

#### Acceptance Criteria

1. THE Test_Suite SHALL hold one Template_Test that runs cfn-lint on the Stack_Template for the Stack_Region in the real Config's `[aws].region`, and fails when cfn-lint reports any error or warning.
2. THE Template_Test SHALL run on the CPU with no network and no AWS credentials.
3. THE Stack_Template SHALL turn off no cfn-lint check, so the Builder fixes each cfn-lint finding in the Stack_Template.
4. THE Test_Suite SHALL pass after this spec, with the new `[aws].stack_name` added to `UNCHANGED_SECTIONS` in `tests/test_config.py`.
5. THE Test_Suite SHALL hold no other new test and no property test for this spec, by the user's choice for this spec, even though the Rules_Steering asks each task to write the task's own tests and asks for property tests.
6. THE Builder SHALL make no AWS call while carrying out this spec, because the Read_Profile doesn't exist until the Account_Owner has deployed the Stack and the Operators have set up the profiles.
7. THE Builder SHALL leave deploying and updating the Stack to the Account_Owner.
8. THE Builder SHALL name, list and describe no Other_Resource in the code, the docs, the commits or the chat.
