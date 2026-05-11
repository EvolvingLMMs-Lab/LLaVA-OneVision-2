      // Theme toggle logic
      // If user has explicitly picked a theme before, honor it.
      // Otherwise leave data-theme unset so CSS @media (prefers-color-scheme)
      // can follow the browser/OS setting automatically.
      var savedTheme = localStorage.getItem('llava-onevision-2-theme');
      if (savedTheme) {
        document.documentElement.dataset.theme = savedTheme;
      }
      document.addEventListener("DOMContentLoaded", function() {
        var dots = document.querySelectorAll('.theme-dot');
        function highlight(theme) {
          dots.forEach(function (d) {
            if (d.dataset.theme === theme) d.classList.add('active');
            else d.classList.remove('active');
          });
        }
        function applyTheme(theme) {
          document.documentElement.dataset.theme = theme;
          localStorage.setItem('llava-onevision-2-theme', theme);
          highlight(theme);
        }
        if (savedTheme) {
          highlight(savedTheme);
        } else {
          // System-driven mode: highlight the dot that matches current OS scheme.
          var prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
          highlight(prefersDark ? 'dark' : 'blue');
        }
        dots.forEach(function (dot) {
          dot.addEventListener('click', function () {
            applyTheme(this.dataset.theme);
          });
        });
      });

      // Language toggle
      var currentLang = 'en';
      var langToggle = document.getElementById('lang-toggle');
      function applyLang(lang) {
        currentLang = lang;
        document.body.classList.toggle('lang-zh', lang === 'zh');
        if (langToggle) langToggle.textContent = lang === 'zh' ? 'EN' : '中文';
      }
      if (langToggle) {
        langToggle.addEventListener('click', function () {
          applyLang(currentLang === 'zh' ? 'en' : 'zh');
        });
      }

      // Mobile TOC toggles (both views)
      document.querySelectorAll('.mobile-toc-toggle').forEach(function (toggle) {
        toggle.addEventListener('click', function () {
          toggle.classList.toggle('open');
          var content = toggle.nextElementSibling;
          if (content) content.classList.toggle('open');
        });
      });

      // Active TOC tracking via IntersectionObserver
      if ('IntersectionObserver' in window) {
        var observer = new IntersectionObserver(function (entries) {
          entries.forEach(function (entry) {
            if (!entry.isIntersecting) return;
            var id = entry.target.id;
            document.querySelectorAll('.toc-list li, .mobile-toc-list li').forEach(function (li) { li.classList.remove('active'); });
            document.querySelectorAll('.toc-list a[href="#' + id + '"], .mobile-toc-list a[href="#' + id + '"]').forEach(function (a) {
              if (a.parentElement) a.parentElement.classList.add('active');
            });
          });
        }, { rootMargin: '-20% 0px -60% 0px', threshold: 0.1 });
        document.querySelectorAll('h3.toc-heading[id]').forEach(function (h) { observer.observe(h); });
      }

      // Pinned first-column shadow + end-of-scroll mask toggling (mobile bench tables)
      document.querySelectorAll('.bench-table-scroll').forEach(function (el) {
        var update = function () {
          el.classList.toggle('is-scrolled', el.scrollLeft > 2);
          var atEnd = el.scrollLeft + el.clientWidth >= el.scrollWidth - 2;
          el.classList.toggle('is-scrolled-end', atEnd);
        };
        el.addEventListener('scroll', update, { passive: true });
        window.addEventListener('resize', update);
        update();
      });

      document.querySelectorAll('.method-figure-svg').forEach(function (el) {
        var update = function () {
          el.classList.toggle('is-scrolled', el.scrollLeft > 4);
        };
        el.addEventListener('scroll', update, { passive: true });
        update();
      });

      // Copy buttons
      document.querySelectorAll('[data-copy-target]').forEach(function (btn) {
        btn.addEventListener('click', function () {
          var target = document.getElementById(btn.getAttribute('data-copy-target'));
          if (!target) return;
          var text = target.innerText;
          navigator.clipboard.writeText(text).then(function () {
            var orig = btn.textContent;
            btn.textContent = currentLang === 'zh' ? '已复制' : 'Copied';
            setTimeout(function () { btn.textContent = orig; }, 1600);
          }).catch(function () {
            btn.textContent = currentLang === 'zh' ? '失败' : 'Error';
          });
        });
      });

      // Highlight code blocks
      if (window.hljs) {
        document.querySelectorAll('pre code').forEach(function (block) {
          window.hljs.highlightElement(block);
        });
      }

      var scrollTopBtn = document.getElementById('scroll-top');
      if (scrollTopBtn) {
        var toggleScrollTop = function () {
          if (window.scrollY > 480) scrollTopBtn.classList.add('visible');
          else scrollTopBtn.classList.remove('visible');
        };
        window.addEventListener('scroll', toggleScrollTop, { passive: true });
        scrollTopBtn.addEventListener('click', function () {
          window.scrollTo({ top: 0, behavior: 'smooth' });
        });
        toggleScrollTop();
      }

document.querySelectorAll('.bench-row td:first-child').forEach(function (td) {
  td.style.cursor = 'pointer';
  td.addEventListener('click', function (e) {
    var btn = td.querySelector('.bench-expand');
    if (!btn) return;
    var row = td.closest('.bench-row');
    var slug = row.getAttribute('data-bench');
    var detail = document.querySelector('.bench-detail[data-bench="' + slug + '"]');
    if (!detail) return;
    var isOpen = row.getAttribute('data-expanded') === 'true';
    row.setAttribute('data-expanded', isOpen ? 'false' : 'true');
    detail.setAttribute('data-open', isOpen ? 'false' : 'true');
    btn.setAttribute('aria-expanded', isOpen ? 'false' : 'true');
  });
});

/* ============================================================ */
/* Demo gallery — tab switching + synced video playback         */
/* ============================================================ */
(function () {
  var tabs = document.querySelectorAll('.demo-tab');
  var panes = document.querySelectorAll('.demo-pane');
  tabs.forEach(function (tab) {
    tab.addEventListener('click', function () {
      var target = tab.getAttribute('data-demo-tab');
      tabs.forEach(function (t) {
        var isActive = t === tab;
        t.classList.toggle('active', isActive);
        t.setAttribute('aria-selected', isActive ? 'true' : 'false');
      });
      panes.forEach(function (p) {
        p.classList.toggle('active', p.id === 'demo-pane-' + target);
      });
      if (target !== 'video') {
        document.querySelectorAll('[data-demo-video]').forEach(function (v) {
          try { v.pause(); } catch (e) {}
        });
      }
    });
  });

  var subtabs = document.querySelectorAll('.demo-subtab');
  var subpanes = document.querySelectorAll('.demo-subpane');
  subtabs.forEach(function (tab) {
    tab.addEventListener('click', function () {
      var target = tab.getAttribute('data-spatial-tab');
      subtabs.forEach(function (t) {
        var isActive = t === tab;
        t.classList.toggle('active', isActive);
        t.setAttribute('aria-selected', isActive ? 'true' : 'false');
      });
      subpanes.forEach(function (p) {
        p.classList.toggle('active', p.id === 'demo-subpane-' + target);
      });
    });
  });

  document.querySelectorAll('[data-demo-carousel]').forEach(function (carousel) {
    var track   = carousel.querySelector('.demo-carousel-track');
    var slides  = carousel.querySelectorAll('.demo-slide');
    var prevBtn = carousel.querySelector('[data-demo-prev]');
    var nextBtn = carousel.querySelector('[data-demo-next]');
    var dots    = carousel.querySelectorAll('.demo-carousel-dot');
    var counter = carousel.querySelector('.demo-carousel-counter');
    if (!track || !slides.length) return;
    var current = 0;
    var total = slides.length;

    function pauseSlide(slide) {
      slide.querySelectorAll('video').forEach(function (v) {
        try { v.pause(); } catch (e) {}
      });
    }
    function playSlide(slide) {
      var pane = carousel.closest('.demo-pane');
      if (pane && !pane.classList.contains('active')) return;
      slide.querySelectorAll('video').forEach(function (v) {
        var p = v.play();
        if (p && typeof p.catch === 'function') p.catch(function () {});
      });
    }
    function update() {
      track.style.transform = 'translateX(-' + (current * 100) + '%)';
      dots.forEach(function (d, i) { d.classList.toggle('active', i === current); });
      if (counter) counter.innerHTML = '<strong>' + (current + 1) + '</strong> / ' + total;
      if (prevBtn) prevBtn.disabled = current === 0;
      if (nextBtn) nextBtn.disabled = current === total - 1;
      slides.forEach(function (slide, i) {
        if (i === current) playSlide(slide); else pauseSlide(slide);
      });
    }
    function go(idx) {
      if (idx < 0 || idx >= total) return;
      current = idx;
      update();
    }
    if (prevBtn) prevBtn.addEventListener('click', function () { go(current - 1); });
    if (nextBtn) nextBtn.addEventListener('click', function () { go(current + 1); });
    dots.forEach(function (dot, i) {
      dot.addEventListener('click', function () { go(i); });
    });

    /* keyboard nav when carousel is focused/hovered */
    carousel.setAttribute('tabindex', '0');
    carousel.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowLeft')  { e.preventDefault(); go(current - 1); }
      if (e.key === 'ArrowRight') { e.preventDefault(); go(current + 1); }
    });

    /* per-slide: click any video to toggle, sync playback across the 3 views */
    slides.forEach(function (slide) {
      var videos = slide.querySelectorAll('video');
      var syncing = false;
      videos.forEach(function (v) {
        v.addEventListener('click', function () {
          if (v.paused) {
            videos.forEach(function (x) {
              try { x.currentTime = v.currentTime; } catch (e) {}
              var p = x.play();
              if (p && typeof p.catch === 'function') p.catch(function () {});
            });
          } else {
            videos.forEach(function (x) { try { x.pause(); } catch (e) {} });
          }
        });
        v.addEventListener('play', function () {
          if (syncing) return;
          syncing = true;
          videos.forEach(function (x) {
            if (x !== v && x.paused) {
              try { x.currentTime = v.currentTime; } catch (e) {}
              var p = x.play();
              if (p && typeof p.catch === 'function') p.catch(function () {});
            }
          });
          syncing = false;
        });
        v.addEventListener('pause', function () {
          if (syncing) return;
          syncing = true;
          videos.forEach(function (x) {
            if (x !== v && !x.paused) {
              try { x.pause(); } catch (e) {}
            }
          });
          syncing = false;
        });
      });
    });

    /* autoplay current slide when carousel scrolls into view */
    if ('IntersectionObserver' in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            playSlide(slides[current]);
          } else {
            slides.forEach(pauseSlide);
          }
        });
      }, { threshold: 0.3 });
      io.observe(carousel);
    }

    update();
  });

  /* When user switches main tabs back to Video, resume current slide */
  tabs.forEach(function (tab) {
    tab.addEventListener('click', function () {
      if (tab.getAttribute('data-demo-tab') === 'video') {
        document.querySelectorAll('[data-demo-carousel]').forEach(function (c) {
          var current = c.querySelector('.demo-carousel-dot.active');
          var idx = current ? Array.prototype.indexOf.call(c.querySelectorAll('.demo-carousel-dot'), current) : 0;
          var slide = c.querySelectorAll('.demo-slide')[idx];
          if (!slide) return;
          slide.querySelectorAll('video').forEach(function (v) {
            var p = v.play();
            if (p && typeof p.catch === 'function') p.catch(function () {});
          });
        });
      }
    });
  });
})();
