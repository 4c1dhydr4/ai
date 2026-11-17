
from django.contrib import admin
from django.urls import path, include


urlpatterns = [
    path("", include("core.urls")),
    path("admin/", admin.site.urls),
    path("classifier/", include("classifier.urls")),
    path("chat/", include("chat.urls")),
    path("rag/", include("rag.urls")),
]
