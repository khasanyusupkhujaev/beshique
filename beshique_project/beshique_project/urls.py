from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('beshique_app.urls')),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

urlpatterns += [
    path("robots.txt", TemplateView.as_view(template_name="robots.txt", content_type="text/plain")),
    path('sitemap.xml', TemplateView.as_view(template_name="sitemap.xml", content_type='application/xml')),
]