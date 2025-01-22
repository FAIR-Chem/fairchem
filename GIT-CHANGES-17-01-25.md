Synchronizing your git history after large file removal on January 17, 2025
---------------------------------------------------------------------------

As a result from our migration into a monorepo which kept the git history from the legacy Open Catalyst repository,
the `FAIRChem` repository was 10MB of files combined with ~350MB of git history for files that no longer exist, but
were still present in the git history.

We removed those files in order to reduce the repository size, but that resulted in changed commit hashes. If you are
currently developing or planning to develop please do the following:

**Do not force push anything or your PR will be rejected**

1. Backup your current repo if you have local changes
   ```bash
       rsync -av current_repo backup_of_current_repo
   ```
2. Migrating a branch (without local changes)
   If you do not have local changes in the branch you wish to keep, ontop of those already on the GitHub (remote)
   > :warning: This will erase all local changes
   ```bash
      git reset --hard origin/branch_name
   ```
3. Migrating a branch (with local changes)
   We need to first unstage any commits that are not present on the remote. To do this use https://github.com/FAIR-Chem/fairchem/tree/main
   to find what the last commit was to the remote branch (the hash wont match, match the commit message). Find the last
   matching commit in your local history, use git log to find this. Once you have the hash from git log, unstage these commits using,
   ```bash
     git reset --soft last_commit_on_remote
   ```
   Then stash the changes, hard reset to the remote, and pop the changes from stash
   ```bash
      git stash
      git reset --hard origin/branch_name
      git stash pop
   ```

If you encounter any issues please open an issue.
