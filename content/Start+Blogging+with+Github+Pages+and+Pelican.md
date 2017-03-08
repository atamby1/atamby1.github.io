Title: Start Blogging with GitHub Pages and Pelican
Date: 2016-11-18
Category: Data Science
Tags: Pelican, GitHub Pages
Author: Avinash TAMBY

For those interested in starting a blog using GitHub Pages and Pelican, here's a brief step-by-step guide on how to get started using a Mac.

But first, I want to go over why I wanted to start a blog, and why I decided to use GitHub Pages and Pelican.

At one of the first data science meetups I went to, the guest lecturer said something that kind of stuck with me: "The absolute worst way to learn something is to take a class on it. The absolute best way to learn something is to teach a class about it." Now, I don't know if I whole-heartedly agree that the worst way to learn something is to take a class on it, but I definitely do agree with the fact that the best way to learn something is to teach it. Being able to organize your thoughts and explain topics that may be relatively complex to someone who has never had any exposure to the topics before really helps drive down concepts that you may have thought you understood, but didn't truly understand.

So, I've decided to start this blog to try to organize my thoughts in a clear and comprehendible manner so that others and my future self can read through and learn something new or review things they already know, but hopefully get a stronger understanding of them.

Today, I'm going to write about how to start a blog. I decided to use Pelican as my static site generator because it's written in Python and this is meant to be a data science blog. I think it's really convenient to use Pelican because it supports the ```Markdown``` format, so I can just write posts in Jupyter Notebook that includes code for projects that I decide to share and everything will automatically be formatted properly. It's also easy to generate the website using a simple CLI (command line interface) tool. Finally, a lot of people have already built really cool themes that you can use for your blog, or if you're really ambitious, you can build your own theme.

[Click here](http://docs.getpelican.com/en/stable/#) to learn more about pelican and [here](http://www.pelicanthemes.com) if you want to take a look at some of the public themes you can base your blog on.

Next, why use GitHub Pages? Well, using GitHub Pages is really a no-brainer if you're already using GitHub to share your projects and code. It's pretty much a free way to host your website and it's really easy to take projects you already have on GitHub and (using Pelican) turn it into a blog post.

Setting Pelican up with GitHub Pages is pretty straightforward, and once you're set up, posting new blog posts and sharing projects is a breeze. So, let's go over how to set up your blog.

Before you get started, you're going to need to download a few packages, so open up Terminal and type in the following commands:

```
pip install pelican
pip install markdown
pip install ghp-import
```

Next, you're going to want to head on over to your GitHub profile and create a new repository with the name 'USERNAME.github.io' where USERNAME is your own GitHub username.

Next, you'll want to clone this repository onto your computer. I decided to clone my repository into Home/Sites, but you can clone this repository wherever you want. I assume your root directory is your home folder, so simply typing ```cd``` should get you into the home folder, then typing ```cd Sites/``` should get you into the Sites folder

So, type the following commands in Terminal:

```
cd
cd Sites/
git clone git@github.com:USERNAME/USERNAME.github.io.git MyBlog
```

You can change ```MyBlog``` into whatever you want the name of this folder to be.

Now that you've cloned your repository, you can cd into MyBlog and run pelican-quickstart to get started.

```
cd MyBlog
pelican-quickstart
```

Now, pelican-quickstart will ask you a series of questions to get your site up. Just hitting return without typing anything will assume you want the default setting. **I want to highlight that you want your URL prefix to be https://USERNAME.github.io ! It should not start with just http://!**

```
> Where do you want to create your new web site? [.] 
> What will be the title of this web site? YOUR TITLE
> Who will be the author of this web site? YOUR NAME
> What will be the default language of this web site? [en] 
> Do you want to specify a URL prefix? e.g., http://example.com   (Y/n) Y
> What is your URL prefix? (see above example; no trailing slash) https://USERNAME.github.io
> Do you want to enable article pagination? (Y/n) Y
> How many articles per page do you want? [10] 
> Do you want to generate a Fabfile/Makefile to automate generation and publishing? (Y/n) Y
> Do you want an auto-reload & simpleHTTP script to assist with theme and site development? (Y/n) Y
> Do you want to upload your website using FTP? (y/N) N
> Do you want to upload your website using SSH? (y/N) N
> Do you want to upload your website using Dropbox? (y/N) N
> Do you want to upload your website using S3? (y/N) N
> Do you want to upload your website using Rackspace Cloud Files? (y/N) N
> Do you want to upload your website using GitHub Pages? (y/N) Y
> Is this your personal page (username.github.io)? (y/N) Y
Done. Your new project is available at /path/to/website
```

Now, you're ready to create your first sample blog post!

Just to test it out, I recommend you just make quick 'Hello World!' post just to make sure everything works properly. Then, you can delete it and make a real post once you have everything set up properly.

After pelican-quickstart finishes, you should have new folders and files your MyBlog folder. Go into the content folder and create a Markdown file by typing the following in Terminal:

```
cd content
touch first-post.md
open first-post.md
```

Now, an empty file will pop in a text editor where you can write your blog post.

Just to make a very quick post, I suggest you type in the following in your first-post.md file:

```
Title: My First Post
Date: 2016-11-20 19:00
Category: Example

Hello world.
```

Now you can go ahead and save that file and exit out of it. Next, in the Terminal type the following:

```
make html
make serve
```

This will serve a local version of what your site will look like so that you can preview it before publishing it to the public. It will be visible at localhost:8000 by default. You can view your website here: http://localhost:8000/

If everything looks good on your website, go back to Terminal and press control + C (aka ^c) to stop the local server. Now you're ready to publish!

Type the following into Terminal:
```
make github
```

Now your website is public and you should be able to view by going to https://USERNAME.github.io !

At this point, I encourage you to look through different themes and personalize your blog a bit. Happy blogging!
