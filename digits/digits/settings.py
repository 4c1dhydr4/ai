import os
from pathlib import Path
from decouple import config
from django.utils.csp import CSP

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = config('SECRET_KEY')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = config('DEBUG', cast=bool)

ALLOWED_HOSTS = ['*'] if DEBUG else config('ALLOWED_HOSTS', cast=lambda v: [s.strip() for s in v.split(',')])


# Application definition

DJANGO_APPS = (
    # 'daphne',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
)

THIRD_PARTY_APPS = (
    # 'channels',
    # 'rest_framework',
    # 'django_celery_results',
)

LOCAL_APPS =  (
    'classifier',
    'chat',
)

INSTALLED_APPS = THIRD_PARTY_APPS + DJANGO_APPS + LOCAL_APPS

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'digits.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'digits.wsgi.application'

# Just for initial development, then will switch to PostgreSQL
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/6.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'America/Lima'

USE_I18N = True

USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = ((BASE_DIR/'static'),)
STATIC_ROOT = BASE_DIR/'s3'

MAILERS = {
    'default': {
        'BACKEND': 'django.core.mail.backends.console.EmailBackend',
    },
}

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
        'mail_admins': {
            'level': 'ERROR',
            'class': 'django.utils.log.AdminEmailHandler',
            'include_html': True,
        }
    },
    'root': {
        'handlers': ['console'],
        'level': 'WARNING',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': os.getenv('DJANGO_LOG_LEVEL', 'WARNING'),
            'propagate': False,
        },
    },
}

# CSP
SECURE_CSP = {
    'default-src': [CSP.SELF, CSP.NONCE],
    'style-src': [CSP.SELF, CSP.UNSAFE_INLINE],
    'style-src-elem': [CSP.SELF, 'https://fonts.googleapis.com', 'https://cdn.jsdelivr.net', CSP.UNSAFE_INLINE],
    'font-src': [CSP.SELF, 'data:', 'https://fonts.googleapis.com', 'https://cdn.jsdelivr.net', 'https://fonts.gstatic.com', CSP.NONCE],
    'script-src': [CSP.SELF, CSP.NONCE],
    'img-src': [CSP.SELF, "https:", 'data:', 'blob:', CSP.NONCE],
    'media-src': [CSP.SELF, "https://example.s3.us-east-2.amazonaws.com", CSP.NONCE],
    'frame-src': [CSP.SELF, '*.google.com', CSP.NONCE]
}

# Permission Policy
PERMISSIONS_POLICY = {
    "accelerometer": ["self"],
    "autoplay": ["self"],
    "camera": ['self'],
    "display-capture": ["self"],
    "encrypted-media": ["self"],
    "fullscreen": ["self"],
    "geolocation": ["self"],
    "gyroscope": ["self"],
    "interest-cohort": ["self"],
    "magnetometer": ["self"],
    "microphone": ["self"],
    "midi": ["self"],
    "payment": ["self"],
    "usb": ["self"],
    "web-share": ['self'],
}

SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_HSTS_SECONDS = 3600