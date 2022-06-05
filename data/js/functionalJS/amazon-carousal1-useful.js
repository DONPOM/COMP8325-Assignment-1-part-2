
      P.when('ready').execute(function() {
        var sidebarbtf = document.getElementById('sidebarbtf');
        var sidebarhidden = document.getElementById('sidebarhidden');
        if (sidebarbtf != null && sidebarhidden != null) {
          sidebarbtf.innerHTML = sidebarhidden.innerHTML;
        }
      });
      P.when('A','ready').execute(function(A) {
        var firstCarousel = A.$('.feed-carousel').first().addClass('first-carousel');
      });
    