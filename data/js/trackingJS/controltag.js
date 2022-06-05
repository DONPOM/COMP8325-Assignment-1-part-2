
/* Controltag Loader for Guardian */
(function() {
    function debugLog(msg) {
        var isDebug = /kxdebug=(1|true)/.test(location);
        if (isDebug && typeof window.console === 'object' && typeof console.info === 'function') {
            console.info(msg);
        }
    }

    function loadCT(url, callback) {
        debugLog('Loading Krux control tag.');
        var ct_element = document.createElement('script');
        ct_element.async = true;
        ct_element.src = url;

        ct_element.onload = ct_element.onreadystatechange = function() {
            var state = ct_element.readyState;
            if (!callback.done && (!state || /loaded|complete/.test(state))) {
                callback.done = true;
                callback();
            }
        };

        // Fetch the first script element, so we can insert the
        // controltag before it. There *must* be at least one
        // script element, or this code would never be called
        var sibling = document.getElementsByTagName('script')[0];
        sibling.parentNode.insertBefore(ct_element, sibling);
    };

    function loadConfig() {


        debugLog('Loading krux configuration.');

        var config = {
            "partner_segment_map": {},
            "context_terms": [],
            "tags": [{
                "content": "<script type=\"text/javascript\">\r\n(function(require) {\r\n  var store = require('store');\r\n  var _ = require('util');  \r\n  var segments;\r\n  \r\n  function handleUserDataResponse(r) {\r\n    // Handle the response as usual.\r\n    segments.handleUserDataResponse(r);\r\n    \r\n    // Override localStorage.kxuser with a case-insensitive value for DFP premium.\r\n    // This will get read by the interchange direct snippet for Krux.user\r\n    // and Krux.dfpKeyValues\r\n    if(r.kuid_long) {\r\n      store.set('user', r.kuid_long, store.DAYS*30);\r\n    }\r\n  }\r\n  \r\n  // Our own fetch uses our own handleResponse\r\n  function fetch() {\r\n    var options = {\r\n      url: _.get('config').services.userData,\r\n      data: {pubid: _.get('pubid')},\r\n      callback: 'kxjsonp_userData',\r\n      done: handleUserDataResponse\r\n    };\r\n    if(!segments.readTechFromStore()) {\r\n      options.data.technographics = 1;\r\n    }\r\n    require('http').jsonp(options);  \r\n  }\r\n  \r\n  \r\n  if(!store.get('segWait')) {\r\n    // disable segments.fetch and use our own\r\n    _.set('segWait', 1, store.MINUTES*5);\r\n    _.onOnce('dom:load', fetch);\r\n    _.fire('user_data_fetch_scheduled');\r\n  }\r\n  \r\n  // Finally, require segments as usual.\r\n  segments = require('segments');\r\n}(Krux.require));\r\n</script>",
                "target": "",
                "target_action": null,
                "require": "",
                "docwrite": null,
                "template_replacement": 1,
                "execution_results": null,
                "tier": 2,
                "internal": 1,
                "content_type": "html",
                "timing": "asap",
                "method": "document",
                "type": "publisher",
                "id": 35371,
                "name": "Krux Load Segments v2"
            }, {
                "content": "<img src=\"//adadvisor.net/adscores/g.js?sid=9212244187&_kdpid=2111c0af-fc3a-446f-ab07-63aa74fbde8e\" />",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "template_replacement": 1,
                "execution_results": null,
                "tier": 1,
                "internal": 22,
                "content_type": "html",
                "freq_cap": 3,
                "timing": "onload",
                "method": "document",
                "type": "data provider",
                "id": 41999,
                "name": "Neustar AdAdvisor S2S provider tag"
            }, {
                "content": "<script>\r\n(function(){\r\n  var kuid = Krux('get', 'user');\r\n  if (kuid) {\r\n    Krux('require:http').pixel({\r\n      url: \"//p.acxiom-online.com/pixel/sci\",\r\n      data: {\r\n          uid: kuid,\r\n          _kdpid: 'f19b7626-3732-4dcc-bac5-8d2c937dae9a',\r\n          pid: 3021\r\n      }});\r\n  }\r\n  })();\r\n</script>",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "template_replacement": 1,
                "execution_results": null,
                "tier": 1,
                "internal": 21,
                "content_type": "html",
                "freq_cap": 3,
                "timing": "onload",
                "method": "document",
                "type": "data provider",
                "id": 42000,
                "name": "Acxiom S2S provider tag"
            }, {
                "content": "<script>\r\n  Krux('require:http').pixel({\r\n    url: \"//s.ixiaa.com/digi/C726AB29-0470-440B-B8D2-D552CED3A3DC/a.gif\"\r\n  });\r\n</script>",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "template_replacement": 1,
                "execution_results": null,
                "tier": 1,
                "internal": 28,
                "content_type": "html",
                "freq_cap": 3,
                "timing": "onload",
                "method": "document",
                "type": "data provider",
                "id": 42001,
                "name": "IXI Digital Open Market provider tag"
            }, {
                "content": "<script>\r\n// Blank\r\n</script>",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "template_replacement": 1,
                "execution_results": null,
                "tier": 1,
                "internal": 19,
                "content_type": "html",
                "freq_cap": 3,
                "timing": "onload",
                "method": "document",
                "type": "data provider",
                "id": 42002,
                "name": "Nielsen (via Exelate) provider tag"
            }, {
                "content": "<script>\r\n\r\n(function() {\r\n    (new Image()).src = \"//apiservices.krxd.net/um?partner=vdna&r=\" + window.location.protocol + \"//e.visualdna.com/conversion&_kdpid=8f95f20d-4acf-43fc-9832-3f5174a166cc\";\r\n})();\r\n\r\n</script>",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "template_replacement": 1,
                "execution_results": null,
                "tier": 1,
                "internal": 12,
                "content_type": "html",
                "freq_cap": 3,
                "timing": "onload",
                "method": "document",
                "type": "data provider",
                "id": 42003,
                "name": "Visual DNA provider tag"
            }, {
                "content": "<script>\r\n(function(){\r\n  var kuid = Krux('get', 'user');\r\n  if (kuid) {\r\n    Krux('require:http').pixel({\r\n      url: \"//loadm.exelator.com/load\",\r\n      data: {\r\n          _kdpid: 'e4942ff0-4070-4896-a7ef-e6a5a30ce9f9',\r\n          buid: kuid,\r\n          p: '204',\r\n          g: '270',\r\n          j: '0'\r\n      }});\r\n  }\r\n  })();\r\n</script>",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "template_replacement": 1,
                "execution_results": null,
                "tier": 1,
                "internal": 11,
                "content_type": "html",
                "freq_cap": 3,
                "timing": "onload",
                "method": "document",
                "type": "data provider",
                "id": 42004,
                "name": "eXelate Media provider tag"
            }, {
                "content": "<script type=\"text/javascript\">Krux('social.init');</script>",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "internal": 1,
                "template_replacement": 1,
                "execution_results": null,
                "tier": 1,
                "library_tag_config": {},
                "content_type": "html",
                "timing": "onload",
                "method": "document",
                "type": "library",
                "id": 35367,
                "name": "Krux Track Social"
            }, {
                "content": "<script>\r\n// this tag is intentionally blank\r\n</script>",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "template_replacement": 1,
                "execution_results": null,
                "tier": 1,
                "internal": 6,
                "content_type": "html",
                "freq_cap": 3,
                "timing": "onload",
                "method": "document",
                "type": "data provider",
                "id": 35368,
                "name": "Technographic Data provider tag"
            }, {
                "content": "",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "template_replacement": 1,
                "execution_results": null,
                "tier": 1,
                "internal": 14,
                "content_type": "html",
                "freq_cap": 3,
                "timing": "onload",
                "method": "document",
                "id": 35369,
                "name": "Krux Geographic Data provider tag"
            }, {
                "content": "<script>\r\n// Using UrlPath 1 to produce page attribute url_path_1\r\nKrux('scrape', { \"page_attr_url_path_1\": {url_path: \"1\"}});\r\n// Using UrlPath 2 to produce page attribute url_path_2\r\nKrux('scrape', { \"page_attr_url_path_2\": {url_path: \"2\"}});\r\n//Setsection/subsection\r\n  Krux('set',{\r\n    section:Krux('get','page_attr_url_path_1'),\r\n    subsection:Krux('get','page_attr_url_path_2') \r\n  });\r\n</script>",
                "target": "",
                "target_action": null,
                "require": "",
                "docwrite": null,
                "template_replacement": 1,
                "execution_results": {
                    "onloadSafe": true,
                    "docwrite": false,
                    "listeners": []
                },
                "tier": 1,
                "internal": false,
                "content_type": "html",
                "timing": "onload",
                "method": "document",
                "type": "publisher",
                "id": 41179,
                "name": "section, subsection"
            }, {
                "content": "<script>\r\nKrux('scrape', { 'page_attr_pageid': {js_global: \"guardian.config.page.pageId\"}});\r\nKrux('scrape', { 'page_attr_section': {js_global: \"guardian.config.page.section\"}});\r\nKrux('scrape', { 'page_attr_edition': {js_global: \"guardian.config.page.edition\"}});\r\nKrux('scrape', { 'page_attr_keywordIds': {js_global: \"guardian.config.page.keywordIds\"}});\r\nKrux('scrape', { 'page_attr_contentType': {js_global: \"guardian.config.page.contentType\"}});\r\nKrux('scrape', { 'page_attr_authorIds': {js_global: \"guardian.config.page.authorIds\"}});\r\n</script>",
                "target": "",
                "target_action": null,
                "require": "",
                "docwrite": null,
                "template_replacement": 1,
                "execution_results": {
                    "onloadSafe": true,
                    "docwrite": false,
                    "listeners": []
                },
                "tier": 1,
                "internal": false,
                "content_type": "html",
                "timing": "onload",
                "method": "document",
                "type": "publisher",
                "id": 41180,
                "name": "DTC NextGen"
            }, {
                "content": "<script type='text/javascript' \r\n        src='//pixel.mathtag.com/sync/js?sync=auto'>\r\n</script>",
                "target": "",
                "target_action": null,
                "require": "",
                "docwrite": null,
                "template_replacement": 1,
                "execution_results": null,
                "tier": 1,
                "internal": 1,
                "content_type": "html",
                "freq_cap": 3,
                "timing": "onload",
                "method": "document",
                "type": "publisher",
                "id": 39151,
                "name": "MediaMath UserMatching"
            }
            ],
            "segments": [],
            "publisher": {
                "uuid": "2196ddf0-947c-45ec-9b0d-0a82fb280cb8",
                "id": 12310,
                "name": "Guardian"
            },
            "controltag_options": {
                "async": "true",
                "render": true
            },
            "site": {
                "id": 18680,
                "name": "Guardian NextGen"
            },
            "dnt": null,
            "params": {
                "connector_host": "connector.krxd.net",
                "control_tag_load_sync": "false",
                "disable_http_compression": false,
                "remove_kxhead": true,
                "supertag_requires_approval": false,
                "control_tag_namespace": null,
                "control_tag_stats_prefix": null,
                "beacon_host": "beacon.krxd.net",
                "control_tag_pixel_throttle": "100",
                "user_id_cookie": null,
                "max_segments": null,
                "no_pii": 0,
                "recommend": false,
                "revenue_optimization": false,
                "context_terms": "false",
                "jslog_host": "jslog.krxd.net",
                "site_level_supertag_config": "site",
                "max_slot_time": 1000,
                "services_host": "apiservices.krxd.net",
                "capture_leakage": true,
                "client_side_storage": "localStorage,cookie",
                "userData_host": "apiservices.krxd.net",
                "control_tag_version": "stable",
                "capture_js_errors": "true",
                "datatag_version": "3"
            },
            "services": {
                "impression": "//beacon.krxd.net/ad_impression.gif",
                "stats": "//apiservices.krxd.net/stats",
                "log": "//jslog.krxd.net/jslog.gif",
                "userData": "//apiservices.krxd.net/user_data/segments/3",
                "um": "//apiservices.krxd.net/um",
                "optout": "//beacon.krxd.net/optout_check",
                "pixel": "//beacon.krxd.net/pixel.gif",
                "contentConnector": "//connector.krxd.net/content_connector/",
                "is_optout": "//beacon.krxd.net/optout_check",
                "set_optin": "//apiservices.krxd.net/consumer/optin",
                "social": "//beacon.krxd.net/social.gif",
                "set_optout": "//apiservices.krxd.net/consumer/optout",
                "data": "//beacon.krxd.net/data.gif",
                "event": "//beacon.krxd.net/event.gif"
            },
            "geo": {},
            "realtime_segments": [],
            "confid": "JVZiE3vn"
        };

        // Wrap in a function and comment, then toString the func and replace
        // everything that's not the beginning or end of a JSON object, so we get a
        // string that's either empty or JSON.
        var esiGeo = String(function() {
            /*
                  <esi:include src="/geoip_esi"/>
                */
        }).replace(/^.*\/\*[^{]+|[^}]+\*\/.*$/g, '');

        if (esiGeo) {
            config.geo = esiGeo;
        }

        Krux('config', config);

    };

    // ControlTag simply crashes when opening a website from Twitter browser on
    // iOS, temporarily disable CT when Twitter for iPhone is detected until they
    // fix this (works fine with other twitter client though)
    if (/Twitter for iPhone/.test(window.navigator.userAgent || '')) 
        return;

    loadCT( "//cdn.krxd.net/ctjs/controltag.js.f99670595ccd46c65a517672ea749179", loadConfig );
})();

