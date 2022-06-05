<!--

window.$Nav && $Nav.declare('config.pageType', 'Gateway');

window.$Nav && $Nav.declare('config.dynamicMenuUrl', '/gp/navigation/ajax/dynamic-menu.html');

window.$Nav && $Nav.declare('config.dismissNotificationUrl',
  '/gp/navigation/ajax/dismissnotification.html');

window.$Nav && $Nav.declare('config.enableDynamicMenus', true);

window.$Nav && $Nav.declare('config.isInternal', false);

window.$Nav && $Nav.declare('config.isRecognized', true);

window.$Nav && $Nav.declare('config.transientFlyoutTrigger', '#nav-transient-flyout-trigger');

window.$Nav && $Nav.declare('config.subnavFlyoutUrl',
  '/gp/navigation/ajax/subnav-flyout');

window.$Nav && $Nav.declare('config.recordEvUrl',
  '/gp/navigation/ajax/recordevent.html');
window.$Nav && $Nav.declare('config.recordEvInterval', 15000);
window.$Nav && $Nav.declare('config.sessionId', '191-1574419-8369751');
window.$Nav && $Nav.declare('config.requestId', '1BB039VX1XJ9CTCMZHE6');

window.$Nav && $Nav.declare('config.readyOnATF', true);

window.$Nav && $Nav.declare('config.dynamicMenuArgs',
  {"rid":"1BB039VX1XJ9CTCMZHE6","isPrime":0,"weblabs":"42812:T1","primeMenuWidth":310});

window.$Nav && $Nav.declare('config.signOutText',
  "Not Muhammad? Sign Out");

window.$Nav && $Nav.declare('config.yourAccountPrimeURL',
  'https://www.amazon.com/gp/css/order-history/utils/first-order-for-customer.html/ref=ya_prefetch_beacon?ie=UTF8&s=191-1574419-8369751');

window.$Nav && $Nav.declare('config.yourAccountPrimeHover',
  true);

window.$Nav && $Nav.declare('config.searchBackState',
  {});










if (window.amznJQ && amznJQ.available) {
  amznJQ.available('jQuery', function() {
    if (!jQuery.isArray) {
      jQuery.isArray = function(o) {
        return Object.prototype.toString.call(o) === "[object Array]";
      };
    }
  });
}

    if (typeof uet == 'function') {
      uet('bb', 'iss-init-pc', {wb: 1});
    }

    if (!window.$SearchJS && window.$Nav) {
      window.$SearchJS = $Nav.make('sx');
    }

  
  var opts = {
      host: "completion.amazon.com/search/complete"
    , marketId: "1"
    , searchAliases: ["aps", "stripbooks", "popular", "apparel", "electronics", "sporting", "garden", "videogames", "toys-and-games", "jewelry", "digital-text", "digital-music", "prime-digital-music", "watches", "grocery", "hpc", "instant-video", "prime-instant-video", "shop-instant-video", "baby-products", "office-products", "software", "magazines", "tools", "automotive", "misc", "industrial", "mi", "pet-supplies", "digital-music-track", "digital-music-album", "mobile", "mobile-apps", "movies-tv", "music-artist", "music-album", "music-song", "stripbooks-spanish", "electronics-accessories", "pantry", "photo", "audio-video", "computers", "furniture", "kitchen", "audible", "audiobooks", "beauty", "shoes", "arts-crafts", "appliances", "gift-cards", "pets", "outdoor", "lawngarden", "collectibles", "financial", "wine", "fine-art", "fashion", "fashion-womens", "fashion-womens-clothing", "fashion-womens-jewelry", "fashion-womens-shoes", "fashion-womens-watches", "fashion-womens-handbags", "fashion-mens", "fashion-mens-clothing", "fashion-mens-jewelry", "fashion-mens-shoes", "fashion-mens-watches", "fashion-girls", "fashion-girls-clothing", "fashion-girls-jewelry", "fashion-girls-shoes", "fashion-girls-watches", "fashion-boys", "fashion-boys-clothing", "fashion-boys-jewelry", "fashion-boys-shoes", "fashion-boys-watches", "fashion-baby", "fashion-baby-boys", "fashion-baby-girls", "fashion-luggage", "3d-printing", "tradein-aps", "local-services", "video-shorts"]
    , isDoCtw: 0
    , pageType: "Gateway"
    , requestId: "1BB039VX1XJ9CTCMZHE6"
    , keydownTriggeredWeblabs: []
    , displayTriggeredWeblabs: []
    , isDdInT3: 0
    , isDdInT1: 0
    , isJpOrCn: 0
    , isUseAuiIss: 0
  };

  var issOpts = {
      fallbackFlag: 1
    , isDigitalFeaturesEnabled: 1
    , isWayfindingEnabled: 1
    , dropdown: "select.searchSelect"
    , departmentText: "in {department}"
    , suggestionText: "Search suggestions"
    , isTriggerIssOnClick: 0
    , imeEnh: 0
    , xcatSuggestionImprovementFlag: 2
    , isShowPromotionInfoInIss: 0
    , np: 0
  };
  

  if (opts.isUseAuiIss === 1) {
    $Nav.when('sx.iss').run('iss-mason-init', function(iss){
      var issInitObj = buildIssInitObject(opts, issOpts, true);
      new iss.IssParentCoordinator(issInitObj);

      tryInitClientTriggeredWeblabs(issInitObj);
    });
  } else if (window.$SearchJS) {
    
    var iss;

    // BEGIN Deprecated globals
    var issHost = opts.host
      , issMktid = opts.marketId
      , issSearchAliases = opts.searchAliases
      , updateISSCompletion = function() { iss.updateAutoCompletion(); };
    // END deprecated globals

    
    
    
    $SearchJS.when('jQuery', 'search-js-autocomplete-lib').run('autocomplete-init', initializeAutocomplete);
    $SearchJS.when('canCreateAutocomplete').run('createAutocomplete', createAutocomplete);

    
    if (opts.isDdInT3) {
      $SearchJS.when('search-js-autocomplete').run('autocomplete-dd-init', function(){ mergeBTFDropdown(); });
    }

    if (opts.isDdInT1) {
      $SearchJS.when('search-js-autocomplete').run('autocomplete-dd-init', function(){ searchDropdown(); });
    }

  } // END conditional for window.$SearchJS

  
  
  function initializeAutocomplete(jQuery) {
    
    var issInitObj = buildIssInitObject(opts, issOpts);

    tryInitClientTriggeredWeblabs(issInitObj);
  } // END initializeAutocomplete

  
  
  function tryInitClientTriggeredWeblabs(issInitObj) {
    
    if (opts.isDoCtw) {
      $SearchJS.importEvent('search-csl');
      $SearchJS.when('search-csl').run('autocomplete-csl-init', function delegateToInitSearchCsl(searchCSL) { initSearchCsl( searchCSL, issInitObj ); } );
    } else {
      $SearchJS.declare('canCreateAutocomplete', issInitObj);
    }
  }

  
  
  function initSearchCsl(searchCSL, issInitObject) {
    searchCSL.init(opts.pageType, (window.ue && window.ue.rid) || opts.requestId);

    
    var keydownCtw = opts.keydownTriggeredWeblabs;
    var displayCtw = opts.displayTriggeredWeblabs;

    
    issInitObject.doCTWKeydown = function(e) {
        for (var i = 0; i < keydownCtw.length; i++) {
          searchCSL.addWlt(keydownCtw[i].call ? keydownCtw[i](e) : keydownCtw[i]);
        }
      };

    issInitObject.doCTWDisplay = function() {
        for (var i = 0; i < displayCtw.length; i++) {
          searchCSL.addWlt(displayCtw[i].call ? displayCtw[i]() : displayCtw[i]);
        }
      };

    $SearchJS.declare('canCreateAutocomplete', issInitObject);
  } // END initSearchCsl

  
  
  function createAutocomplete(issObject) {
    iss = new AutoComplete(issObject);

    $SearchJS.publish('search-js-autocomplete', iss);

    logMetrics();
  } // END createAutocomplete

  
  
  function buildIssInitObject(opts, issOpts, isNewIss) {
    var issInitObj = {
        src: opts.host
      , mkt: opts.marketId
      , aliases: opts.searchAliases
      , fb: issOpts.fallbackFlag
      , isDigitalFeaturesEnabled: issOpts.isDigitalFeaturesEnabled
      , isWayfindingEnabled: issOpts.isWayfindingEnabled
      , deptText: issOpts.departmentText
      , sugText: issOpts.suggestionText
      , ime: opts.isJpOrCn
      , mktid: opts.marketId
      , qs: opts.isJpOrCn
      , deepNodeISS: {
          searchAliasAccessor: function() {
            return (window.SearchPageAccess && window.SearchPageAccess.searchAlias()) ||
                   jQuery('select.searchSelect').children().attr('data-root-alias');
          },
          searchAliasDisplayNameAccessor: function() {
            return (window.SearchPageAccess && window.SearchPageAccess.searchAliasDisplayName());
          }
        }
    };

    // If we aren't using the new ISS then we need to add these properties
    if (!isNewIss) {
      issInitObj.dd = issOpts.dropdown; // The element with id searchDropdownBox doesn't exist in C.
      issInitObj.imeEnh = issOpts.imeEnh;
      issInitObj.imeSpacing = issOpts.imeSpacing;
      issInitObj.xcatSuggestionImprovementFlag = issOpts.xcatSuggestionImprovementFlag;
      issInitObj.isShowPromotionInfoInIss = issOpts.isShowPromotionInfoInIss;
      issInitObj.isNavInline = 1;
      issInitObj.triggerISSOnClick = issOpts.triggerIssOnClick
      issInitObj.sc = 1;
      issInitObj.np = issOpts.np;
    }

    return issInitObj;
  } // END buildIssInitObject

  
  function logMetrics() {
    if (typeof uet == 'function' && typeof uex == 'function' ) {
      uet('be', 'iss-init-pc', {wb: 1});
      uex('ld', 'iss-init-pc', {wb: 1});
    }
  } // END logMetrics




    window.amznJQ && amznJQ.declareAvailable('navbarInline');
    window.$Nav && $Nav.declare('nav.inline');

    window.amznJQ && amznJQ.available('jQuery', function() {
        amznJQ.available('navbarJS-beacon', function(){});
    });

(function (i) {
i.onload = function() {window.uet && uet('ne')};
i.src = window._navbarSpriteUrl;
}(new Image()));

window.$Nav && $Nav.declare('config.autoFocus', false);


window.$Nav && $Nav.declare('config.responsiveTouchAgents', ["ieTouch"]);

window.$Nav && $Nav.declare('config.responsiveGW',true);

window.$Nav && $Nav.declare('config.newFlyouts',true);

window.$Nav && $Nav.declare('config.velocityFlyoutToggling', true);
window.$Nav && $Nav.declare('config.velocityFlyoutThreshold', 40);

window.$Nav && $Nav.declare('config.twoClickFlyouts',false);

window.$Nav && $Nav.declare('config.sslTriggerType','pageReady');
window.$Nav && $Nav.declare('config.sslTriggerRetry',0);

window.$Nav && $Nav.declare('config.doubleCart',false);

window.$Nav && $Nav.declare('config.csm',false);


window.$Nav && $Nav.declare('config.signInTooltip',false);

window.$Nav && $Nav.declare('config.isPrimeMember',false);

window.$Nav && $Nav.declare('config.primeTooltip',false);

window.$Nav && $Nav.declare('config.carnac',false);


window.$Nav && $Nav.declare('config.ewc',false);


window.$Nav && $Nav.declare('config.blackbelt', true);

window.$Nav && $Nav.declare('config.searchapiEndpoint',false);

    window._navbar = window._navbar || {};
    window._navbar.browsepromos = window._navbar.browsepromos || {};
    
 _navbar.browsepromos['desktop-shopall-android'] = {"width":540,"promoType":"wide","vertOffset":0,"horizOffset":0,"height":522,"image":"http://g-ecx.images-amazon.com/images/G/01/Appstore/Flyout/flyout-bb-talesFromDeepSpace._V320395556_.png"}; 
 _navbar.browsepromos['desktop-shopall-automotive-industrial'] = {"width":541,"promoType":"wide","vertOffset":0,"horizOffset":0,"height":524,"image":"http://g-ecx.images-amazon.com/images/G/01/img14/automotive/flyout/12132_automotive_blackfriday_flyout_us_black-white-belt._V320438274_.png"}; 
 _navbar.browsepromos['desktop-shopall-books'] = {"width":540,"promoType":"wide","vertOffset":0,"horizOffset":0,"height":522,"image":"http://g-ecx.images-amazon.com/images/G/01/img14/books/flyout/12942_books_product-alert_flyout_us_blackbelt._V320466420_.png"}; 
 _navbar.browsepromos['desktop-shopall-clothing-shoes-jewelry'] = {"width":540,"promoType":"wide","vertOffset":0,"horizOffset":0,"height":515,"image":"http://g-ecx.images-amazon.com/images/G/01/AMAZON_FASHION/2014/WAYFINDING/EDITORIAL/HOL_2/GATEWAY/FLYOUTS/FO_BLACKBELT_1_BFW_65off_all._V320130658_.png"}; 
 _navbar.browsepromos['desktop-shopall-cloud-drive'] = {"width":540,"promoType":"wide","vertOffset":0,"horizOffset":0,"height":522,"image":"http://g-ecx.images-amazon.com/images/G/01/digital/adrive/images/gno/us_cd_prime_bb_gno_2._V320633312_.png"}; 
 _navbar.browsepromos['desktop-shopall-credit'] = {"width":540,"promoType":"wide","vertOffset":0,"horizOffset":0,"height":522,"image":"http://g-ecx.images-amazon.com/images/G/01/credit/img14/PLCC/plcc6MonthFinancing_Blackbelt_flyout._V320078765_.png"}; 
 _navbar.browsepromos['desktop-shopall-digital-music'] = {"width":540,"promoType":"wide","vertOffset":0,"horizOffset":0,"height":522,"image":"http://g-ecx.images-amazon.com/images/G/01/digital/music/merch/holiday2014/AllIsBright/AIB_110314_flyout_Blackbelt._V320371864_.png"}; 
 _navbar.browsepromos['desktop-shopall-electronics-computers'] = {"width":540,"promoType":"wide","vertOffset":0,"horizOffset":0,"height":522,"image":"http://g-ecx.images-amazon.com/images/G/01/img14/ce-accessories/egg/reciprocals/gateway/9658_electronics_EGGGatewayUS-FLYOUT_BLACKBELT_03._V320253137_.png"}; 
 _navbar.browsepromos['desktop-shopall-grocery-health-beauty'] = {"width":541,"promoType":"wide","vertOffset":0,"horizOffset":0,"height":524,"image":"http://g-ecx.images-amazon.com/images/G/01/img14/grocery/flyout/grocery_gno_bf_flyout_us_black-white-belt_A._V320603030_.png"}; 
 _navbar.browsepromos['desktop-shopall-home-garden-tools'] = {"width":540,"promoType":"wide","vertOffset":0,"horizOffset":0,"height":522,"image":"http://g-ecx.images-amazon.com/images/G/01/img14/patio-lawn-garden/flyout/14293_plg_experience-the-flavor-b_flyout_us_black-white-belt._V320700531_.png"}; 
 _navbar.browsepromos['desktop-shopall-instant-video'] = {"width":540,"promoType":"wide","vertOffset":0,"horizOffset":0,"height":522,"image":"http://g-ecx.images-amazon.com/images/G/01/digital/video/merch/GNOflyout/2014-GNO_GG_S1_FullBleed-FinalPNG_BlackBelt._V320340757_.png"}; 
 _navbar.browsepromos['desktop-shopall-movies-music-games'] = {"width":540,"promoType":"wide","vertOffset":0,"horizOffset":0,"height":522,"image":"http://g-ecx.images-amazon.com/images/G/01/img14/movies-tv/flyout/14174_Movies-TV_flyout_blackbelt_Monday-v2._V320719381_.png"}; 
 _navbar.browsepromos['desktop-shopall-sports-outdoors-t1'] = {"width":540,"promoType":"wide","vertOffset":0,"horizOffset":0,"height":522,"image":"http://g-ecx.images-amazon.com/images/G/01/img14/sports-outdoors/flyout/od_holiday14_acsphotoshoot_blackbelt_2._V320634324_.png"}; 
 _navbar.browsepromos['desktop-shopall-toys-kids-baby'] = {"width":541,"promoType":"wide","vertOffset":0,"horizOffset":0,"height":524,"image":"http://g-ecx.images-amazon.com/images/G/01/img14/toys/flyout/13888_toys_bkfriday-deals_flyout_us_black-white-belt._V320719841_.png"}; 
 _navbar.browsepromos['desktop-shopall_k-fire-phone'] = {"width":540,"promoType":"wide","vertOffset":0,"horizOffset":0,"height":515,"image":"http://g-ecx.images-amazon.com/images/G/01/kindle/merch/2014/campaign/fp/h/FP_Blackbelt_H_540x515._V327130929_.png"}; 
 _navbar.browsepromos['desktop-shopall_k-fire-tablet'] = {"width":540,"promoType":"wide","vertOffset":0,"horizOffset":0,"height":516,"image":"http://g-ecx.images-amazon.com/images/G/01/kindle/merch/2014/campaign/G7/Family/fire-tablets-GNO-blackbelt-540x516._V325759433_.png"}; 
 _navbar.browsepromos['desktop-shopall_k-fire-tv'] = {"width":540,"promoType":"wide","vertOffset":"-20","horizOffset":"0","height":516,"image":"http://g-ecx.images-amazon.com/images/G/01/kindle/merch/2014/campaign/kb/app/AFTV-GNO-blackbelt-540x516._V321877984_.png"}; 
 _navbar.browsepromos['desktop-shopall_k-reader'] = {"width":540,"promoType":"wide","vertOffset":0,"horizOffset":0,"height":515,"image":"http://g-ecx.images-amazon.com/images/G/01/kindle/merch/2014/campaign/G7/Family/kindle-GNO-blackbelt-540x516_v6._V320414234_.png"}; 


    window.$Nav && $Nav.declare('config.browsePromos', window._navbar.browsepromos);


window.$Nav && $Nav.declare('configComplete');

-->