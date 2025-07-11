from django import forms

class WaitlistForm(forms.Form):
    email = forms.EmailField(label='', widget=forms.EmailInput(attrs={'placeholder': 'name@framer.com', 'class': 'w-full'}))