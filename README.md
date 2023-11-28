# btx
Miscellaneous functions for aiding analysis during LCLS experiments.

## Documentation

*Coming soon...*

see *Doxygen* documentation here: https://lcls-users.github.io/btx/

## Contributing
This repository is the centralized location to discuss and share calibration files, documentation, scripts and code for SFX analysis.

See [Discussions](https://github.com/lcls-users/btx/discussions) above for a forum experience.
Contribute or check [Wiki](https://github.com/lcls-users/btx/wiki) above for tutorials and tips, shared experience, etc.

For contributing code and script, please don't push to the main branch directly. Rather, work in a specific branch for the task at hand, and make a pull request (PR) so it can be safely merged into the main branch after the person you assign as a reviewer does the code review (if necessary).

More importantly, have fun! If you have any questions, feel free to post a new issue or just reach out directly to someone else on the team.

## Etiquette

Before you start working on contribution, please make sure your local main branch is up-to-date. Create a new branch from it, add or edit, then push a pull request.

To create your local repository (from this GitHub hosted remote repository):
```
$ git clone git@github.com:lcls-users/btx.git
```
To update your local main branch:
```
$ git checkout main
$ git pull
```
Now you can create a new branch that will be up to date with the current (remote) main branch:
```
$ git checkout -b my-new-branch
$ ... do stuff ..
$ git add <the files that you have been working on>
$ git commit -m "Hey everyone, I have been working on those files to do this and that"
$ git push origin my-new-branch
```
At this point, checkout the repo on GitHub and create a Pull Request (like [this one](https://github.com/apeck12/btx/pull/11)). 
Once merged, the main branch will be updated with your work!
