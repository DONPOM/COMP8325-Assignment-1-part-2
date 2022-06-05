
    $(window).load(function () {
      var dataAttribute = Math.max(window.innerWidth, window.innerHeight) <= 640 ?
        'mobile' : 'original';
      
      $('.j-lazyload-hp').lazyload({
        effect : 'fadeIn',
        data_attribute: dataAttribute,
      });
    });
    // Slideshare Object

      slideshare_object.favorites = {};
      slideshare_object.favorites.create_url = '/favorite/create';
      slideshare_object.favorites.delete_url = '/favorite/delete';
      slideshare_object.favorites.user_favorites = '/favorite/get_favorites';

    slideshare_object.bind_favorites('body');


    var gaCat = "homepage_loggedin_created_on_ss";

    slideshare_object.ga(gaCat, 'pageload', null, null, true);

    $(document).ready(function() {
      if (getUrlVar('from', window.location.href) === "logout") {
        var expiry = new Date();
        expiry.setHours(expiry.getHours() + 4);
        cookie("autologin_disabled", true, { path: '/', expires: expiry });
      }

      // Resize handler

      var slideshowRatio = 225 / 300;
      var $slideshows = $(".ss-slideshow-thumbnail, .ss-conference-thumbnail");
      var $slideshowThumbnails = $slideshows.find(".link-bg-img");

      var resizeHandler = function() {
        $slideshows.each(function(index) {
          var $this = $(this);
          var $slideshowThumbnail = $this.find(".link-bg-img");

          $slideshowThumbnail.height(Math.floor($slideshowThumbnail.width() * slideshowRatio));

          if ($this.hasClass('ss-slideshow-thumbnail') && $this.width() < 250) {
            $this.addClass("ss-small-thumbnail");
          } else {
            $this.removeClass("ss-small-thumbnail");
          }
        });
      };

      $(window).resize(function() { resizeHandler(); });

      resizeHandler();

      // Favorites update

      if (slideshare_object.user.loggedin) {
        slideshare_object.favoritesStatusUpdate();
      }

      // Modal player

      var carouselModalPlayer = new CarouselModalPlayer();

      // Bind list page modal share

      if (slideshare_object.is_mobile) {
        var mobile_modal_share = new MobileModalShare({
          shareBtn: '.j-share.mobile-share',
          parentEl: 'body',
          el: '.j-mobileShare'
        });
      } else {
        var listPageModalShare = new ListPageModalShare({
          btnShare: '.j-share.desktop-share',
          parentEl: 'body',
          el: '#homepage-modal-share'
        });
      }

      // LI connect

      if (slideshare_object.user.loggedin){
        var login_source = "homepage_loggedin";
        var connectRoute = "bind_account";
      }else{
         var login_source = "homepage_loggedout";
         var connectRoute = "connect";
      }
      var config = {
        'connectButtonClass': ".j-connect-li-hp",
        'state': "0566dd89e712dd795978639a47f651",
        'login_source': login_source,
        'connectRoute' :  connectRoute
      };

      ssliconnect = new LIConnect(config);

      /* Update status of the follow action
       *
       * Parameters:
       * elem <selector> the action-btn to be toggled
       * category url of the category to update
       */

      var followStatusUpdate = function(el, category){
        var $el = $(el);
        var isFollowing = $el.children().hasClass('fa-check');

        var config = {
          'url':  isFollowing ? "/follow/delete" : "/follow/create",
          'data': {
            user_id : slideshare_object.user ? slideshare_object.user.id : '',
            category : category
          }
        };

        $el.children().toggleClass('fa-check fa-plus');

        $.ajax({
          url: config.url,
          data: config.data,
          dataType: 'json',
          type: 'POST',
          success: function(response) {
            if(response.success){
              slideshare_object.ga(gaCat, isFollowing ? 'unfollow_topic_click' : 'follow_topic_click', category);
            }
          },
          error: function(data) {
            $el.children().toggleClass('fa-check fa-plus');
          }
        });
      };

      $('.j-action-follow').on('click', function(e) {
        var $this = $(this);
        var category = $this.data('category');

        if (slideshare_object.user.loggedin){
          e.preventDefault();
          followStatusUpdate(this, category);
        } else {
          var eventConfig = {
            "event" : "follow_category",
            "data" : { "category" : category}
          };

          slideshare_object.addAfterLoginEvent(eventConfig);
          window.location = '/login?login_source=topics.follow_category&from_source=/topics';
        }
      });

      // Ads
      $('#hp-ad-1').html(slideshare_object.ads_config.ads_homepage_right2_fill);

      // Mobile Promotions
      if (mobile_util.isIOS8() && !mobile_util.isIPad() && mobile_util.portraitMode()) {
        var iosTopBanner = {
          'currentPage': 'foundation_homepage',
          'promoName': 'ios_top_banner',
          'promoSelector': '#ios-top-banner',
          'lazyloadClass': '.j-lazyload',
          'downloadSelector': '.j-store',
          'closedSelector': '.j-close',
          'launchOnInit': true,
          'navbarSelector': '#main-nav',
          'contentSelector': '.wrapper'
        };

        var iosTopBanner = new MobilePromo(iosTopBanner);
      }
      else if (mobile_util.isAndroid()) {
        if (mobile_util.isAndroidPhone()) {
          var promoName = 'android_phone_splash';
          var promoSelector = '#android_splash_container.phone-splash';
        }
        else {
          var promoName = 'android_tablet_splash';
          var promoSelector = '#android_splash_container.tablet-splash';
        }

        var androidSplashConfig = {
          'currentPage': 'foundation_homepage',
          'promoName': promoName,
          'promoSelector': promoSelector,
          'lazyloadClass': '.j-splash_lazy',
          'downloadSelector': '.j-store_btn',
          'closedSelector': '.j-close_btn',
          'launchOnInit': true
        };

        var androidSplash = new MobilePromo(androidSplashConfig);
      }
    });
  