import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

os.environ['DJANGO_SETTINGS_MODULE'] = 'beshique_project.settings'

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
