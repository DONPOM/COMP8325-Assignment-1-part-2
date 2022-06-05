$(function() {
  // Foundation init

  $(document).foundation();

  /*** GA ***/

  // Init GA

  (function() {
    var ga = document.createElement('script'); ga.type = 'text/javascript'; ga.async = true;
    ga.src = ('https:' == document.location.protocol ? 'https://ssl' : 'http://www') + '.google-analytics.com/ga.js';
    var s = document.getElementsByTagName('script')[0]; s.parentNode.insertBefore(ga, s);
  })();

  // Tracking helper

  $('body').on('click', '[data-ga-cat]', function() {
    var gaCategory = $(this).data('ga-cat');
    var gaAction = $(this).data('ga-action');
    var gaLabel = $(this).data('ga-label');
    var gaValue = $(this).data('ga-value');
    var gaNonInteractive = $(this).data('ga-noninteractive') || false;

    if (gaCategory && gaAction) {
      slideshare_object.ga(gaCategory, gaAction, gaLabel, gaValue, gaNonInteractive);
    }
  });

  /*** APP init ***/

  slideshare_object.setLanguageSelector();

  slideshare_object.autosuggestTop();

  slideshare_object.lazyloadThumbnails();

  $.ajaxSetup({
    headers: {
      'X-CSRF-Token': $('meta[name="csrf-token"]').attr('content')
    }
  });

  // Mobile search bar

  var toggleSearchBar = function() {
    var $body = $('body');
    var $mobileSearch = $('#main-nav-mobile-search');

    if ($mobileSearch.is(':visible')) {
      $mobileSearch.hide().find('input').blur();
    } else {
      $mobileSearch.slideDown().find('input').focus();
    }
  };

  $('body').on('click', '.j-open-mobile-search', function(e) {
    e.stopPropagation();
    e.preventDefault();
    toggleSearchBar();
  });

  $('#main-nav-mobile-search form').on('submit', function(e) {
    $('#main-nav-mobile-search').addClass("searching");
  });

  // Global body click handler

  $('body').on('click', function(e) {
    // Check if click is outside mobile search
    if ($(e.target).closest("#main-nav-mobile-search").length <= 0) {
      if ($('#main-nav-mobile-search').is(':visible')) {
        e.stopPropagation();
        e.preventDefault();
        toggleSearchBar();
      }
    }
  });

  // Makes sure lazyload is triggered when the page loads

  $(window).load(function(){
    var $body = $('body');
    var $main_nav = $('#main-nav');
    
    // Update the html body padding top based on the height of the fixed main_nav
    var updateBodyPaddingTop = function() {
      $body.css('padding-top', $main_nav.height());
    };

    // Determine if the Term of Service banner should be shown and display it
    var initTOSBanner = function() {
      var $tosBanner = $('.j-tos-update-banner');
      var $showText = mobile_util.isMobile() ? $('.j-mobile-text') : $('.j-desktop-text');
      var displayCount = parseInt(mobile_util.getCookie('tos_update_banner_shows'), 10);
      displayCount = isNaN(displayCount) ? 0 : displayCount;
      var numToShow = mobile_util.isMobile() ? 1 : 2;
      if ($tosBanner.length && displayCount < numToShow) {
        $tosBanner.show();
        $showText.show();
        updateBodyPaddingTop();
        mobile_util.setCookie('tos_update_banner_shows', displayCount + 1, 365);

        $tosBanner.find('.j-tos-close').on('click', function(e) {
          $tosBanner.hide();
          $showText.hide();
          updateBodyPaddingTop();
        });
      }
    };

    $(window).resize(function(){
      // Resize the main_nav and body to the correct positions 
      updateBodyPaddingTop();
    });

    // TOS Update Banner; Disabled when not needed
    // initTOSBanner();

    $("html,body, body .wrapper").trigger("scroll");
  });
});
