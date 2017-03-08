#!/usr/bin/env python
# -*- coding: utf-8 -*- #

from __future__ import unicode_literals

PLUGIN_PATHS = ['plugins/pelican-plugins']
PLUGINS = ['i18n_subsites']

#JINJA2_EXTENSIONS = ['gutils.utils.templates.CacheExtension']
#'jinja2.ext.i18n', 
JINJA_ENVIRONMENT = {'extensions': ['jinja2.ext.autoescape', 'jinja2.ext.with_']}

AUTHOR = u'Avinash TAMBY'
SITEURL = u''
SITENAME = u"Avinash's Blog"
SITETITLE = AUTHOR
SITESUBTITLE = u'Data Scientist'
SITEDESCRIPTION = u'%s\'s Thoughts and Writings' % AUTHOR
SITELOGO = u'images/Avinash.jpg'
SITE_THUMBNAIL = u'images/A.png'
FAVICON = SITEURL + u'images/A.png'
BROWSER_COLOR = '#333'

ROBOTS = u'index, follow'

THEME = u'themes/Flex'
PATH = u'content'
TIMEZONE = u'America/New_York'
DEFAULT_LANG = u'en'
OG_LOCALE = u'en_US'

DATE_FORMATS = {
    'en': '%d %B %Y',
}

'''
FEED_ALL_ATOM = 'feeds/all.atom.xml'
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None
'''

USE_FOLDER_AS_CATEGORY = True
MAIN_MENU = True

LINKS = (('portfolio', 'https://atamby1.github.io/portfolio'),)


SOCIAL = (('linkedin', 'https://www.linkedin.com/in/avinashtamby'),
          ('envelope', 'mailto:atamby1@gmail.com'),
          ('github', 'https://github.com/atamby1'),
          )


MENUITEMS = (('Archives', '/archives.html'),
             ('Categories', '/categories.html'),
             ('Tags', '/tags.html'),)

CC_LICENSE = {
    'name': 'Creative Commons Attribution-ShareAlike',
    'version': '4.0',
    'slug': 'by-sa'
}

COPYRIGHT_YEAR = 2017

DEFAULT_PAGINATION = 10

STATUSCAKE = {
    'trackid': 'test-test',
    'days': 7,
    'rumid': 1234,
}

RELATIVE_URLS = False

FEED_ALL_ATOM = 'feeds/all.atom.xml'
CATEGORY_FEED_ATOM = 'feeds/%s.atom.xml'

DELETE_OUTPUT_DIRECTORY = False

DEFAULT_PAGINATION = 5
SUMMARY_MAX_LENGTH = 150

#DISQUS_SITENAME = "test-test"
#GOOGLE_ANALYTICS = "UA-XXXXXX-X"
#ADD_THIS_ID = 'ra-XX3242XX'

USE_LESS = True
