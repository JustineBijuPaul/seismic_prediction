(function(html) {

    "use strict";

    html.className = html.className.replace(/\bno-js\b/g, '') + ' js ';

    const tl = anime.timeline({
        easing: 'easeInOutCubic',
        duration: 800,
        autoplay: false
    })
    .add({
        targets: '#loader',
        opacity: 0,
        duration: 1000,
        begin: function(anim) {
            window.scrollTo(0, 0);
        }
    })
    .add({
        targets: '#preloader',
        opacity: 0,
        complete: function(anim) {
            document.querySelector("#preloader").style.visibility = "hidden";
            document.querySelector("#preloader").style.display = "none";
        }
    })
    .add({
        targets: '.s-header',
        translateY: [-100, 0],
        opacity: [0, 1]
    }, '-=200')
    .add({
        targets: ['.s-intro .text-pretitle', '.s-intro .text-huge-title'],
        translateX: [100, 0],
        opacity: [0, 1],
        delay: anime.stagger(400)
    })
    .add({
        targets: '.circles span',
        keyframes: [
            {opacity: [0, .3]},
            {opacity: [.3, .1], delay: anime.stagger(100, {direction: 'reverse'})}
        ],
        delay: anime.stagger(100, {direction: 'reverse'})
    })
    .add({
        targets: '.intro-social li',
        translateX: [-50, 0],
        opacity: [0, 1],
        delay: anime.stagger(100, {direction: 'reverse'})
    })
    .add({
        targets: '.intro-scrolldown',
        translateY: [100, 0],
        opacity: [0, 1]
    }, '-=800');

    const ssPreloader = function() {

        const preloader = document.querySelector('#preloader');
        if (!preloader) return;

        window.addEventListener('load', function() {
            document.querySelector('html').classList.remove('ss-preload');
            document.querySelector('html').classList.add('ss-loaded');

            document.querySelectorAll('.ss-animated').forEach(function(item){
                item.classList.remove('ss-animated');
            });

            tl.play();
        });
    };

    const ssMobileMenu = function() {

        const toggleButton = document.querySelector('.mobile-menu-toggle');
        const mainNavWrap = document.querySelector('.main-nav-wrap');
        const siteBody = document.querySelector("body");

        if (!(toggleButton && mainNavWrap)) return;

        toggleButton.addEventListener('click', function(event) {
            event.preventDefault();
            toggleButton.classList.toggle('is-clicked');
            siteBody.classList.toggle('menu-is-open');
        });

        mainNavWrap.querySelectorAll('.main-nav a').forEach(function(link) {
            link.addEventListener("click", function(event) {

                if (window.matchMedia('(max-width: 800px)').matches) {
                    toggleButton.classList.toggle('is-clicked');
                    siteBody.classList.toggle('menu-is-open');
                }
            });
        });

        window.addEventListener('resize', function() {

            if (window.matchMedia('(min-width: 801px)').matches) {
                if (siteBody.classList.contains('menu-is-open')) siteBody.classList.remove('menu-is-open');
                if (toggleButton.classList.contains("is-clicked")) toggleButton.classList.remove("is-clicked");
            }
        });

    };

    const ssScrollSpy = function() {

        const sections = document.querySelectorAll(".target-section");

        window.addEventListener("scroll", navHighlight);

        function navHighlight() {

            let scrollY = window.pageYOffset;

            sections.forEach(function(current) {
                const sectionHeight = current.offsetHeight;
                const sectionTop = current.offsetTop - 50;
                const sectionId = current.getAttribute("id");

                const navLink = document.querySelector(`.main-nav a[href*="${sectionId}"]`);

                if (navLink && navLink.parentNode) {
                    if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
                        navLink.parentNode.classList.add("current");
                    } else {
                        navLink.parentNode.classList.remove("current");
                    }
                }
            });
        }

    };

    const ssViewAnimate = function() {

        const blocks = document.querySelectorAll("[data-animate-block]");

        window.addEventListener("scroll", viewportAnimation);

        function viewportAnimation() {

            let scrollY = window.pageYOffset;

            blocks.forEach(function(current) {

                const viewportHeight = window.innerHeight;
                const triggerTop = (current.offsetTop + (viewportHeight * .2)) - viewportHeight;
                const blockHeight = current.offsetHeight;
                const blockSpace = triggerTop + blockHeight;
                const inView = scrollY > triggerTop && scrollY <= blockSpace;
                const isAnimated = current.classList.contains("ss-animated");

                if (inView && (!isAnimated)) {
                    anime({
                        targets: current.querySelectorAll("[data-animate-el]"),
                        opacity: [0, 1],
                        translateY: [100, 0],
                        delay: anime.stagger(400, {start: 200}),
                        duration: 800,
                        easing: 'easeInOutCubic',
                        begin: function(anim) {
                            current.classList.add("ss-animated");
                        }
                    });
                }
            });
        }

    };

    const ssSwiper = function() {

        const mySwiper = new Swiper('.swiper-container', {

            slidesPerView: 1,
            pagination: {
                el: '.swiper-pagination',
                clickable: true,
            },
            breakpoints: {
                401: {
                    slidesPerView: 1,
                    spaceBetween: 20
                },
                801: {
                    slidesPerView: 2,
                    spaceBetween: 32
                },
                1201: {
                    slidesPerView: 2,
                    spaceBetween: 80
                }
            }
        });

    };

    const ssLightbox = function() {

        const folioLinks = document.querySelectorAll('.folio-list__item-link');
        const modals = [];

        folioLinks.forEach(function(link) {
            let modalbox = link.getAttribute('href');
            let instance = basicLightbox.create(
                document.querySelector(modalbox),
                {
                    onShow: function(instance) {
                        document.addEventListener("keydown", function(event) {
                            event = event || window.event;
                            if (event.keyCode === 27) {
                                instance.close();
                            }
                        });
                    }
                }
            )
            modals.push(instance);
        });

        folioLinks.forEach(function(link, index) {
            link.addEventListener("click", function(event) {
                event.preventDefault();
                modals[index].show();
            });
        });

    };

    const ssAlertBoxes = function() {

        const boxes = document.querySelectorAll('.alert-box');

        boxes.forEach(function(box){

            box.addEventListener('click', function(event) {
                if (event.target.matches(".alert-box__close")) {
                    event.stopPropagation();
                    event.target.parentElement.classList.add("hideit");

                    setTimeout(function(){
                        box.style.display = "none";
                    }, 500)
                }
            });

        })

    };

    const ssMoveTo = function(){

        const easeFunctions = {
            easeInQuad: function (t, b, c, d) {
                t /= d;
                return c * t * t + b;
            },
            easeOutQuad: function (t, b, c, d) {
                t /= d;
                return -c * t* (t - 2) + b;
            },
            easeInOutQuad: function (t, b, c, d) {
                t /= d/2;
                if (t < 1) return c/2*t*t + b;
                t--;
                return -c/2 * (t*(t-2) - 1) + b;
            },
            easeInOutCubic: function (t, b, c, d) {
                t /= d/2;
                if (t < 1) return c/2*t*t*t + b;
                t -= 2;
                return c/2*(t*t*t + 2) + b;
            }
        }

        const triggers = document.querySelectorAll('.smoothscroll');

        const moveTo = new MoveTo({
            tolerance: 0,
            duration: 1200,
            easing: 'easeInOutCubic',
            container: window
        }, easeFunctions);

        triggers.forEach(function(trigger) {
            moveTo.registerTrigger(trigger);
        });

    };
    
    function visualizeWaveform(data) {
        const ctx = document.getElementById('waveformChart').getContext('2d');
        
        // Create time array
        const timePoints = Array.from({length: data.waveform.length}, (_, i) => i / data.sampling_rate);
        
        // Create markers for earthquakes and aftershocks
        const earthquakePoints = data.earthquake_indices.map(i => ({
            x: i / data.sampling_rate,
            y: data.waveform[i]
        }));
        
        const aftershockPoints = data.aftershock_indices.map(i => ({
            x: i / data.sampling_rate,
            y: data.waveform[i]
        }));
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: timePoints,
                datasets: [{
                    label: 'Waveform',
                    data: data.waveform,
                    borderColor: 'rgb(75, 192, 192)',
                    borderWidth: 1,
                    fill: false
                }, {
                    label: 'Earthquake Events',
                    data: earthquakePoints,
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgb(255, 99, 132)',
                    pointStyle: 'circle',
                    pointRadius: 5,
                    showLine: false
                }, {
                    label: 'Aftershocks',
                    data: aftershockPoints,
                    borderColor: 'rgb(255, 159, 64)',
                    backgroundColor: 'rgb(255, 159, 64)',
                    pointStyle: 'triangle',
                    pointRadius: 4,
                    showLine: false
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time (seconds)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Amplitude'
                        }
                    }
                }
            }
        });
        
        // Display analysis results
        document.getElementById('analysisResults').innerHTML = `
            <h3>Analysis Results:</h3>
            <p>Prediction: ${data.prediction}</p>
            <p>Confidence: ${data.confidence}</p>
            <p>Estimated Magnitude: ${data.magnitude}</p>
            <p>Dominant Frequency: ${data.analysis.dominant_freq.toFixed(2)} Hz</p>
            <p>Peak Amplitude: ${data.analysis.peak_amplitude.toFixed(4)}</p>
            <p>RMS: ${data.analysis.rms.toFixed(4)}</p>
        `;
    }

    (function ssInit() {

        ssPreloader();
        ssMobileMenu();
        ssScrollSpy();
        ssViewAnimate();
        ssSwiper();
        ssLightbox();
        ssAlertBoxes();
        ssMoveTo();

    })();

    document.addEventListener('DOMContentLoaded', function() {
        const form = document.getElementById('upload-form');
        const results = document.getElementById('results');
        const alertBox = document.querySelector('.alert-box');
    
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('seismic-file');
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                displayResults(data);
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred during analysis');
            }
        });
    
        function displayResults(data) {
            results.classList.remove('hidden');
            
            // Display probability
            document.querySelector('.probability').textContent = 
                `${(data.probability * 100).toFixed(1)}%`;
            
            // Display time to event if available
            if (data.hours_to_event) {
                document.querySelector('.time-to-event').textContent = 
                    `${data.hours_to_event.toFixed(1)} hours (${data.confidence} confidence)`;
            }
            
            // Display confidence
            document.querySelector('.confidence').textContent = data.confidence || 'N/A';
            
            // Display precursor information
            document.querySelector('.precursors').textContent = 
                data.has_precursors ? 'Detected' : 'Not Detected';
            
            // Display alert if present
            if (data.alert) {
                alertBox.classList.remove('hidden');
                alertBox.classList.remove('severe', 'warning');
                alertBox.classList.add(data.alert.level.toLowerCase());
                
                alertBox.querySelector('h4').textContent = data.alert.level;
                alertBox.querySelector('.message').textContent = data.alert.message;
                alertBox.querySelector('.action').textContent = data.alert.action_required;
            } else {
                alertBox.classList.add('hidden');
            }
        }
    });

})(document.documentElement);
