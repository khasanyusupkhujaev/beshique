from django.shortcuts import render, redirect
from django.core.mail import send_mail
from .forms import WaitlistForm
from .models import WaitlistEntry

def home(request):
    if request.method == 'POST':
        form = WaitlistForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            created, _ = WaitlistEntry.objects.get_or_create(email=email)

            if created:
                # Send thank-you email
                send_mail(
                    subject="You're on the Beshique wishlist — thank you! ❤️",
                    message="""Hi there,

Thank you for joining the Beshique wishlist — it means the world to us.

We’re currently building our first prototype of the smart baby cradle, and we're working day and night to bring it to life. Beshique will adapt to your baby’s needs with intelligent rocking and calming sounds — giving parents the sleep and peace they deserve.

We'll keep you updated at every key stage: from prototype to production, and finally to launch. You’ll be the first to know when pre-orders open.

Thank you for believing in us this early.

Warmly,  
The Beshique Team
""",
                    from_email="Beshique <beshiqueai@gmail.com>",
                    recipient_list=[email],
                    fail_silently=False,
                )

            return redirect('home')
    else:
        form = WaitlistForm()

    return render(request, 'home.html', {'form': form})


def faq(request):
    return render(request, 'faq.html')


def blogs(request):
    return render(request, 'blogs.html')
