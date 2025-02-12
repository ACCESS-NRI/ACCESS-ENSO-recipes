# Review notebooks from GitHub

You will need to use your GitHub account. 
Repos can have certain permissions so reach out to the repo owner if something isn't working for you. See links at the bottom for more thorough tutorials and information on git.

## Git repos and Pull requests
A repo is a repository which is like a project folder of code, scripts and associated files. These can be hosted remotely so that multiple people can view, use and contribute. GitHub is a common host for remote repositories with the pull request feature to review and for others to add to the project.

<p align="left"><img src="docs/assets_github/clone_button.png" alt="screenshot" width="60%"/></p>

### 1. To start off working you can clone this repo to your workspace in either:

- your local machine if you have your own python environments you use
- on gadi (and use ARE to run the notebooks)
    - `git clone <HTTPS_OR_SSH>`

This means you can:
- run and test the code in your workspace
- create your own branches to edit or add code and maintain your versions 
    - `git checkout -b *<name_your_branch>*` *from updated main branch
- or pull any updates as others work on it.
    - `git pull origin/main` 
- or switch to a different branch to look at their changes
    - `git fetch` to get any updated branches from remote repo (GitHub)
    - `git checkout origin/<name_of_branch>` 

You can also use git extensions and GUIs in desktop, eg. VSCode has a source control panel. It lists repos, you can find git commands from the repo menu and which branch it is on.
<p align="left"><img src="docs/assets_github/vscode_source.png" alt="screenshot" width="50%"/></p>
Click on the branch name and you can select a branch to switch to (after you have done a fetch to get all available from the remote repository)
<p align="left"><img src="docs/assets_github/vscode_branch.png" alt="screenshot" width="60%"/></p>

### 2. Pull requests are created on GitHub 
You can create pull requests on GitHub for your branch of edits to merge into the `main` branch. 
<p align="center"><img src="docs/assets_github/pr_header.png" alt="screenshot" width="60%"/></p>

A new pull request is created here where you select which branch you would like to merge.

These can then be reviewed by others, adding general comments and suggestions to code as well as comments to specific lines. These can be addressed with further edits before it is approved and merged into the `main` branch.
<p align="center"><img src="docs/assets_github/pr_reviews.png" alt="screenshot" width="60%"/></p>

In a PR, you can review changes by going to the *Files changed* tab, where you will see a list of the files modified and can add comments to lines with the blue **+** that appears when you hover over the line numbers.

## Git and Jupyter notebooks
Git doesn't work well with Jupyter notebooks because of all the differences in kernels for cells and running them even though the python code may not have changed.
We have an extension installed for this repo [*ReviewNB*](reviewnb.com) to assist with reviewing notebooks. You will find a link in the pull request.
<p align="center"><img src="docs/assets_github/reviewNB.png" alt="screenshot" width="60%"/></p>

You can then add comments to specific lines in the notebook through ReviewNB similar to GitHub in a pull request.
<p align="center"><img src="docs/assets_github/reviewNB_line.png" alt="screenshot" width="60%"/></p>

As mentioned, git doesn't work well with notebooks, we modify one notebook per branch for review, so one PR per notebook. When we add `commits` to that branch we can more easily make sure that only that notebook is included and reduce the merge conflicts that are likely to arise across different branches. So, only one branch should be live with edits to a notebook, only one person should edit a notebook at a time and commit and push changes before someone else pulls those updates and adds their own edits.

## Links and guides
- there is great documentation and [tutorials from Atlassian](https://www.atlassian.com/git/tutorials)
    - a page on [git commands](https://www.atlassian.com/git/glossary#commands)
    - and working with remote repos: [syncing](https://www.atlassian.com/git/tutorials/syncing) and [pull requests](https://www.atlassian.com/git/tutorials/making-a-pull-request)
