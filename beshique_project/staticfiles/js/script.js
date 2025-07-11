document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

document.addEventListener("DOMContentLoaded", () => {
    const preloader = document.getElementById("preloader");
  
    const MIN_DISPLAY_TIME = 1000; // 1 second
    const start = Date.now();
  
    window.addEventListener("load", function () {
      const elapsed = Date.now() - start;
      const delay = Math.max(0, MIN_DISPLAY_TIME - elapsed);
  
      setTimeout(() => {
        preloader.classList.add("opacity-0");
        setTimeout(() => {
          preloader.style.display = "none";
        }, 500); // matches fade-out transition
      }, delay);
    });
});