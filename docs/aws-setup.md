# AWS setup

The AWS setup is one CloudFormation stack, made from `cloud/aws-setup.yaml`, plus a few steps by hand. The steps below go from a new AWS account to running jobs.

## Who does what

Only the account owner uses the AWS console and CloudShell. They sign in with the new admin login that Step 2 makes, with MFA, and use the root user only where Steps 1, 2 and 6 say. Only the account owner deploys or updates the stack.

Everyone else uses only the AWS CLI on their own laptop, with their own keys, and so does the account owner outside the console.

Kiro never deploys or updates the stack. On a laptop, the guard hook blocks the commands that create or update stacks when Kiro runs them.

Every aws command on a laptop names its profile with `--profile`, never with `AWS_PROFILE`, because AWS keys in environment variables beat `AWS_PROFILE`. Every aws command names its region with `--region`, because an `AWS_REGION` or `AWS_DEFAULT_REGION` environment variable beats the profile's region, even with `--profile`.

The commands in CloudShell are the one exception to `--profile`. They pass only `--region`, because CloudShell runs as the login you signed in to the console with, and it has no named profiles.

The guide uses these placeholders and `config.toml` values:

- `<your-region>` is the region in `[aws].region` in `config.toml`.
- `voxelis-kiroween` is the stack's name, in `[aws].stack_name`.
- `kiroween-read` and `kiroween-runner` are your read profile and your runner profile, in `[aws].read_profile` and `[aws].runner_profile`.
- The project tag is the tag `project` with the value `voxelis-kiroween`, as in `[aws].project_tag`.

A copy that changes the stack's name or the profile names in `config.toml` uses its own names wherever the guide has these.

## What the stack makes

The stack makes a private job bucket, the role and the instance profile that job machines run with, and their security group, which lets nothing in and only HTTPS and HTTP out. It also makes the runner role, which starts and cleans up job machines, the read-only policy for everyday use, and the project group for the people who run jobs.

The job bucket's layout:

- The copies of a job's `input/` and `out/` folders go under `jobs/<job id>/input/` and `jobs/<job id>/out/`. They stay until someone deletes them.
- The copy of the job's `work/` folder goes under `work/<job id>/`. The bucket deletes each file there 60 days after its upload.

The bucket also cancels any upload that's still unfinished 7 days after it started, anywhere in the bucket, and deletes the parts it got.

## Step 1: Move the account to the paid plan

Sign in as the root user, because the admin login only comes in Step 2. In the Billing and Cost Management console, if the account is on AWS's free plan, upgrade it to the paid plan.

An account on the free plan can't use every service, and AWS closes it when its credits run out or after six months.

Leave the account's setting for IAM access to billing as it is.

## Step 2: Make the admin login

Make a new admin login, even if the account already has an IAM user with administrator access, so the setup depends on no other user in the account.

1. Signed in as the root user, open IAM, choose Users, then Create user.
2. Give the user a name that starts with the stack's name, like `voxelis-kiroween-admin`.
3. Give it console access, with a password you choose.
4. Attach AWS's managed `AdministratorAccess` policy directly.
5. Add the tag `project` with the value `voxelis-kiroween`, then create the user.
6. Sign in as the new user, with the sign-in URL on the IAM dashboard, and assign it an MFA device.

Make no access keys for the admin login. Its sign-in URL holds the account id, so keep the URL out of chats.

From here on, use the admin login instead of the root user, except for the budget in Step 6 when the Billing console refuses the admin login.

## Step 3: Check the default VPC

In the VPC console, in `<your-region>`, Your VPCs should list a VPC whose Default VPC column says Yes. If none does, choose Actions, then Create default VPC.

The stack's security group goes in the default VPC, job machines start in its default subnets, and Step 4's create command needs its id. So without a default VPC, the stack fails.

## Step 4: Deploy the stack

You deploy the stack once, from CloudShell. Only the account owner deploys the stack, never Kiro.

1. Sign in with the admin login, pick `<your-region>` in the console's region menu, and open CloudShell. If CloudShell isn't available in `<your-region>`, the console opens it in the nearest region that has it, as AWS's [CloudShell guide](https://docs.aws.amazon.com/cloudshell/latest/userguide/getting-started.html) says. That still works, because each aws command below passes `--region <your-region>`, which beats the region that CloudShell sets for its shell ([Working in AWS Regions](https://docs.aws.amazon.com/cloudshell/latest/userguide/working-with-aws-cloudshell.html#region-selection)).
2. Remove any older copy of the template with the first line below, `rm -f aws-setup.yaml`.
3. Choose Actions, then Upload file, and upload `cloud/aws-setup.yaml` from your copy of the repo. It lands in CloudShell's home folder, where the commands below find it.
4. Run the other three lines: the create command, the wait command, which waits for up to an hour, and the status check, which prints only the stack's status. Look for `CREATE_COMPLETE`.

```bash
rm -f aws-setup.yaml
aws cloudformation create-stack --stack-name voxelis-kiroween --template-body file://aws-setup.yaml --capabilities CAPABILITY_IAM --tags Key=project,Value=voxelis-kiroween --parameters ParameterKey=DefaultVpcId,ParameterValue=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query "Vpcs[0].VpcId" --output text --region <your-region>) --region <your-region> > /dev/null
aws cloudformation wait stack-create-complete --stack-name voxelis-kiroween --region <your-region>
aws cloudformation describe-stacks --stack-name voxelis-kiroween --query "Stacks[0].StackStatus" --output text --region <your-region>
```

The create command looks up the default VPC's id inline, with `describe-vpcs` and the `is-default` filter, and passes it as the stack's one parameter, so the id isn't printed. The create command's own output is the stack's ARN, which holds the account id, so `> /dev/null` drops it. Its errors still show.

If the region has no default VPC, the lookup gives `None`, and the create command fails at once with a parameter validation error, before anything is made. Go back to Step 3, then run the create command again.

The template goes through CloudShell because `--template-body` sends it inside the request, so nothing outside the stack is made or reused. Uploading a template file in the CloudFormation console has CloudFormation keep a copy of the file in an S3 bucket outside the stack, which CloudFormation makes without the project tag, or reuses if the account already has one for the region. So don't upload the template in the console. `--template-body` takes at most 51,200 bytes, and the template stays within that.

The project tag also goes on the stack, as a stack tag, because the project group, the instance profile, the managed policies and the bucket policy take no tags in CloudFormation.

If the create fails, the wait command stops with an error, and the status check prints `ROLLBACK_COMPLETE`. The first failed event on the stack's Events tab in the CloudFormation console says why. Reading it there is fine, because it uploads nothing. Fix the cause, delete the failed stack with the delete commands below, and create it again.

```bash
aws cloudformation delete-stack --stack-name voxelis-kiroween --region <your-region>
aws cloudformation wait stack-delete-complete --stack-name voxelis-kiroween --region <your-region>
```

A failed first create leaves no job bucket behind, because CloudFormation deletes the still-empty bucket when the first create rolls back. So there's nothing to clean up by hand.

## Step 5: Add the people who run jobs

For each person who runs jobs, the account owner included:

1. In IAM, choose Users, then Create user. Give the user a name that starts with the stack's name, like `voxelis-kiroween-<name>`, and leave console access off.
2. Add the user to the group that the stack's `GroupName` output names. The output is on the stack's Outputs tab in the CloudFormation console.
3. Give the user the tag `project` with the value `voxelis-kiroween`, then create it.
4. On the user's Security credentials tab, choose Create access key, for the Command Line Interface.
5. Hand the access key id and the secret access key to that person privately, for example in person or through a password manager.

The keys go only into that person's own AWS CLI files, so never into the repo, a chat, an email or a log.

Adding someone to the group is the account owner's OK for them to run jobs on the account.

The admin login has no access keys, and neither the read profile nor the runner profile in Step 7 uses an admin's keys. So Kiro never holds admin keys.

## Step 6: Set a budget alert

Sign in with the admin login, or with the root user if the Billing console refuses the admin login. In Billing and Cost Management, choose Budgets, then Create budget. Make a monthly cost budget for an amount you pick, with an email alert, and give it the tag `project` with the value `voxelis-kiroween`.

The budget covers the whole account, because a budget for the project tag alone needs the tag turned on as a cost allocation tag, which is an account-wide billing setting.

The alert only sends an email. It stops nothing.

Leave the account's setting for IAM access to billing as it is.

## Step 7: Set up your AWS CLI profiles

Each person who runs jobs, the account owner included, does this once, in their own terminal and not through Kiro. Each command below passes `--profile` and `--region <your-region>`.

1. Install the AWS CLI v2.
2. Make your read profile from your access keys from Step 5:

   ```
   aws configure --profile kiroween-read --region <your-region>
   ```

   It asks for the access key id, the secret access key, the default region (type `<your-region>`) and the output format (`json`).

   If `aws configure` ends by offering to configure AWS skills and the AWS MCP server for your AI coding agent, answer `n`, because Kiro gets AWS access only through your two profiles.

3. Make your runner profile, which assumes the runner role with your read profile's keys. The first line reads the runner role's ARN from the stack's `RunnerRoleArn` output with your read profile, and writes it into your runner profile. In PowerShell:

   ```powershell
   aws configure set role_arn (aws cloudformation describe-stacks --stack-name voxelis-kiroween --query "Stacks[0].Outputs[?OutputKey=='RunnerRoleArn'].OutputValue" --output text --profile kiroween-read --region <your-region>) --profile kiroween-runner --region <your-region>
   ```

   In bash:

   ```bash
   aws configure set role_arn "$(aws cloudformation describe-stacks --stack-name voxelis-kiroween --query "Stacks[0].Outputs[?OutputKey=='RunnerRoleArn'].OutputValue" --output text --profile kiroween-read --region <your-region>)" --profile kiroween-runner --region <your-region>
   ```

   If the inner command prints an error instead, nothing useful is written: fix your read profile and run the line again. Then, in either shell, make your read profile the runner profile's source profile, and set the runner profile's region:

   ```
   aws configure set source_profile kiroween-read --profile kiroween-runner --region <your-region>
   aws configure set region <your-region> --profile kiroween-runner --region <your-region>
   ```

4. Run the two checks. Each prints only the stack's status, like `CREATE_COMPLETE` or `UPDATE_COMPLETE`. The second one shows that your runner profile can assume the runner role.

   ```
   aws cloudformation describe-stacks --stack-name voxelis-kiroween --query "Stacks[0].StackStatus" --output text --profile kiroween-read --region <your-region>
   aws cloudformation describe-stacks --stack-name voxelis-kiroween --query "Stacks[0].StackStatus" --output text --profile kiroween-runner --region <your-region>
   ```

The runner role's ARN holds the account id, so it goes only into your own AWS CLI config file, `~/.aws/config`, or `%USERPROFILE%\.aws\config` on Windows. The `role_arn` line writes it there without printing it.

AWS error messages can hold the account id, so don't paste them into a chat. Don't use `aws sts get-caller-identity` as a check, because it prints the account id.

The runner profile's credentials last an hour, and the AWS CLI renews them from your read profile.

Your read profile's keys can also assume the runner role, because the project group allows that. So the line between the two profiles is Kiro's approval prompt, the spend guard in `cloud/spend.py` and the rules in `.kiro/steering/rules.md`, not IAM.

## Step 8: Ask for GPU quota

The GPU quota is "Running On-Demand G and VT instances", with the quota code `L-DB2E81BA`. It counts vCPUs, per account and per region. A job machine counts 4 vCPUs for an xlarge, 8 for a 2xlarge, 16 for a 4xlarge and 32 for an 8xlarge.

The instance types that job machines use:

| Instance type | vCPUs | GPU | What it's for |
|---|---|---|---|
| g5.xlarge | 4 | 1 NVIDIA A10G | The smoke test |
| g5.8xlarge | 32 | 1 NVIDIA A10G | Fast and default jobs |
| g5.4xlarge | 16 | 1 NVIDIA A10G | Making the machine image, and fast and default jobs where the quota is under 32 vCPUs |
| g6e.4xlarge | 16 | 1 NVIDIA L40S | Quality jobs |
| g6.4xlarge | 16 | 1 NVIDIA L4 | Fallback |
| g5.2xlarge | 8 | 1 NVIDIA A10G | Fallback |
| g6.2xlarge | 8 | 1 NVIDIA L4 | Fallback |
| g6e.2xlarge | 8 | 1 NVIDIA L40S | Fallback |

The runner role refuses every instance type outside this table, so using another type means changing `cloud/aws-setup.yaml` and updating the stack.

A new account starts at 0, so no job can start until AWS approves an increase. The account owner asks for one, signed in with the admin login:

1. Open the Service Quotas console, and pick `<your-region>` in the console's region menu.
2. Choose AWS services, then Amazon Elastic Compute Cloud (Amazon EC2).
3. Choose "Running On-Demand G and VT instances", then Request increase at account level.
4. Ask for 32 vCPUs, so fast and default jobs can run on a g5.8xlarge.

You can read the quota and the requests to raise it on your laptop, with your read profile. The first command below prints the quota's value, a number like `32.0`. The second prints the status and the asked-for value of each request, in lines like `CASE_OPENED 32.0`, and leaves out each request's ARNs and requester.

```
aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --query "Quota.Value" --output text --profile kiroween-read --region <your-region>
aws service-quotas list-requested-service-quota-change-history-by-quota --service-code ec2 --quota-code L-DB2E81BA --query "RequestedQuotas[].[Status,DesiredValue]" --output text --profile kiroween-read --region <your-region>
```

## Opening a shell on a job machine

Install the Session Manager plugin for the AWS CLI. Then, in your own terminal, open a shell with your runner profile, with the job machine's instance id in place of `<instance-id>`:

```
aws ssm start-session --target <instance-id> --profile kiroween-runner --region <your-region>
```

It works only on machines with the project tag. You can end or resume only the shells that your own runner session started. Session Manager ends an idle shell after its idle timeout, 20 minutes unless the account's preferences say otherwise, and every shell ends with its machine.

Kiro doesn't open shells.

## Stack outputs

The stack has these outputs, on its Outputs tab in the CloudFormation console:

| Output | What it holds | Who uses it |
|---|---|---|
| `BucketName` | The job bucket's name | The start command, to upload `input/` and fetch `out/`. The job machines, through the start command. People reading results with their read profile. |
| `JobRoleArn` | The ARN of the role that job machines run as | Our cloud code, where it needs to name the role. Nobody types it. |
| `JobInstanceProfileArn` | The ARN of the instance profile that holds that role | The start command, which starts every job machine with it |
| `RunnerRoleArn` | The ARN of the runner role | Each person's runner profile, set once in Step 7 |
| `SecurityGroupId` | The id of the job machines' security group | The start command |
| `VpcId` | The id of the default VPC that holds the security group | The start command, which picks one of its default subnets |
| `GroupName` | The project group's name | The account owner, adding people in Step 5 |

The ARNs hold the account id, so keep them out of chats and logs.

## Updating the stack

When `cloud/aws-setup.yaml` changes, the account owner updates the stack in CloudShell, opened with the admin login as in Step 4. Pick a name for the change set, of letters, digits and hyphens that starts with a letter, like `update-20261120`, and put it in place of `<name>` in the commands below.

1. Remove the old copy of the template with the line below, then choose Actions, then Upload file, and upload the new `cloud/aws-setup.yaml` from your copy of the repo.

   ```bash
   rm -f aws-setup.yaml
   ```

2. Make the change set. It passes the same stack tag as Step 4's create command, so the stack keeps its project tag whatever CloudFormation does with a change set that leaves the tags out. It passes `UsePreviousValue=true` for the default VPC's id, so the stack keeps the id it was made with. Its own output is the change set's ARN, so `> /dev/null` drops it. Its errors still show.

   ```bash
   aws cloudformation create-change-set --stack-name voxelis-kiroween --change-set-name <name> --template-body file://aws-setup.yaml --capabilities CAPABILITY_IAM --tags Key=project,Value=voxelis-kiroween --parameters ParameterKey=DefaultVpcId,UsePreviousValue=true --region <your-region> > /dev/null
   ```

3. Wait for the change set:

   ```bash
   aws cloudformation wait change-set-create-complete --stack-name voxelis-kiroween --change-set-name <name> --region <your-region>
   ```

4. Review the change set, on the stack's Change sets tab in the CloudFormation console, or with the command below. It prints one line per change, with the change's action, logical id and replacement, like `Modify JobPolicy False`, and `None` as the replacement of a resource the change set adds. It leaves out the physical ids, which for the managed policies are ARNs that hold the account id.

   ```bash
   aws cloudformation describe-change-set --stack-name voxelis-kiroween --change-set-name <name> --query "Changes[].ResourceChange.[Action,LogicalResourceId,Replacement]" --output text --region <your-region>
   ```

5. Look for `JobBucket` with a replacement of `True`, or `Conditional`, which means it may be replaced. A replacement makes a new, empty job bucket, and keeps the old one with its files outside the stack, because of the bucket's `UpdateReplacePolicy: Retain`. If that isn't what you want, delete the change set with the command below instead of executing it, and ask whoever changed the template.

   ```bash
   aws cloudformation delete-change-set --stack-name voxelis-kiroween --change-set-name <name> --region <your-region>
   ```

6. Otherwise, execute the change set, then wait for the update to finish:

   ```bash
   aws cloudformation execute-change-set --stack-name voxelis-kiroween --change-set-name <name> --region <your-region>
   aws cloudformation wait stack-update-complete --stack-name voxelis-kiroween --region <your-region>
   ```

Never use `aws cloudformation update-stack`, which skips the review. IAM changes can take a few seconds to apply after the update.

Kiro never runs any of these commands.

## Removing the setup

The account owner removes the setup. First, in the EC2 console, terminate every job machine that's still running. Job machines are the instances with the project tag. Then, in this order:

1. In IAM, take every person's user out of the project group, the group that the stack's `GroupName` output names, because CloudFormation can't delete a group that still has members.
2. Note the job bucket's name from the stack's `BucketName` output, on its Outputs tab, because the outputs go with the stack. Then delete the stack in CloudShell, opened with the admin login as in Step 4, with the delete commands:

   ```bash
   aws cloudformation delete-stack --stack-name voxelis-kiroween --region <your-region>
   aws cloudformation wait stack-delete-complete --stack-name voxelis-kiroween --region <your-region>
   ```

3. In the S3 console, empty the job bucket, then delete it. Deleting the stack keeps the bucket and its files, because of the bucket's `DeletionPolicy: RetainExceptOnCreate`, while the bucket policy goes with the stack.

The machine images and snapshots that the runner role made aren't part of the stack, so deleting the stack leaves them. They carry the project tag. Deregister them and delete their snapshots in the EC2 console.

The IAM users, the admin login and the budget aren't part of the stack either, so they stay too.

## Another region or another name

A copy that deploys in another region or under another name changes `[aws].region` or `[aws].stack_name` in `config.toml`, and uses those values in Steps 3, 4, 7 and 8 and in the commands after them.

IAM names come from CloudFormation, so copies in two regions of one account don't clash. Each region needs its own default VPC and its own GPU quota.

Two copies in one region of one account would share the project tag, so a second copy goes in another region.
