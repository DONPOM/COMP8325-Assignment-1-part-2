(function() {
    var delayedEmbed = yt.getConfig('DELAYED_EMBED', []);
    var delayedSwfConfig = {"args": {"enablejsapi": 1}, "params": {"allowfullscreen": "false", "allowscriptaccess": "always", "bgcolor": "#FFFFFF"}, "url_v8": "", "html5": false, "url": "\/\/s.ytimg.com\/yts\/swf\/masthead_child-vflRMMO6_.swf", "attrs": {"height": "1", "id": "masthead_child", "width": "1"}, "min_version": "8.0.0", "url_v9as2": ""};
      delayedSwfConfig.fallbackMessage = function(){};
    delayedEmbed.push({
      'container': "masthead_child_div",
      'swf_config': delayedSwfConfig
    });
    yt.setConfig('DELAYED_EMBED', delayedEmbed);
  })();
 yt.setConfig({
        'MASTHEAD_ENCRYPTED_ID': "B7vGbdS4bcE",
        'MASTHEAD_IS_BRANDED': false
      });

(function() {var c = document.getElementById('feed-pyv-container');if (c) {var h = "\u003cscript\u003ewindow.yt = window.yt || {};yt.www = yt.www || {};yt.www.feed = yt.www.feed || {};yt.www.feed.ui = yt.www.feed.ui || {};yt.www.feed.ui.ads = yt.www.feed.ui.ads || {};window.renderHomepagePyv = function(pyv) {writePyvAd();};\u003c\/script\u003e\u003cscript\u003evar renderPyvCallback = parent.yt.www.ads.pyv.pyvHomeAfcCallback;window.writePyvAd = function() {window['google_page_url'] = parent.document.location;window['google_cust_age'] = \"1002\";window['google_cust_gender'] = \"1\";window['google_language'] = \"en\";window['google_loeid'] = \"__loeid__\";window['google_ad_client'] = \"__property_code__\";window['google_ad_channel'] = \"__channels__\";window['google_max_num_ads'] = 1;window['google_ad_output'] = 'js';window['google_ad_type'] = 'text';window['google_only_pyv_ads'] = true;window['google_ad_request_done'] = function(googleAds) {renderPyvCallback(googleAds);};document.write('\u003cscript s\u0072c=\"\/\/pagead2.googlesyndication.com\/pagead\/show_ads.js\"\u003e\u003c\\\/script\u003e');};var pyvCallback = window.renderHomepagePyv;pyvCallback();\u003c\/script\u003e";h = h.replace(/__property_code__/g, c.getAttribute('data-property-code') || '');h = h.replace(/__channels__/g, c.getAttribute('data-channels') || '');h = h.replace(/__loeid__/g, c.getAttribute('data-loeid') || '');yt.setConfig({'PYV_IFRAME_CONTENT': h,'PYV_IFRAME_ID': 'pyv-iframe'});}})();
      yt.setConfig('JS_PAGE_MODULES', 'www/feed');

    yt.setConfig('DISMISS_THROUGH_IT', true);

      yt.setConfig({
        'GUIDE_SELECTED_ITEM': "0qDduQEREg9GRXdoYXRfdG9fd2F0Y2g%3D"
      });

      yt.setConfig({
    'GUIDED_HELP_LOCALE': "en_US",
    'GUIDED_HELP_ENVIRONMENT': "prod"
  });