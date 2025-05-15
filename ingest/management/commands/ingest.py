from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Prints Hello, world!"

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Hello, world!"))
