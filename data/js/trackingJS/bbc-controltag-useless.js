
/* Controltag Loader for BBC */
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
                "content": "<script type=\"text/javascript\">\r\n// DFP premium.\r\n(function(require) {\r\n  var store = require('store');\r\n  var _ = require('util');  \r\n  var segments;\r\n  \r\n  function handleUserDataResponse(r) {\r\n    // Handle the response as usual.\r\n    segments.handleUserDataResponse(r);\r\n    \r\n    // Override localStorage.kxuser with a case-insensitive value for DFP premium.\r\n    // This will get read by the interchange direct snippet for Krux.user\r\n    // and Krux.dfpKeyValues\r\n    if(r.kuid_long) {\r\n      store.set('user', r.kuid_long, store.DAYS*30);\r\n    }\r\n  }\r\n  \r\n  // Our own fetch uses our own handleResponse\r\n  function fetch() {\r\n    var options = {\r\n      url: _.get('url_userData'),\r\n      data: {pubid: _.get('pubid')},\r\n      callback: 'kxjsonp_userData',\r\n      done: handleUserDataResponse\r\n    };\r\n    if(!segments.readTechFromStore()) {\r\n      options.data.technographics = 1;\r\n    }\r\n    require('http').jsonp(options);  \r\n  }\r\n  \r\n  \r\n  if(!store.get('segWait')) {\r\n    // disable segments.fetch and use our own\r\n    _.set('segWait', 1, store.MINUTES*5);\r\n    _.onOnce('dom:load', fetch);\r\n    _.fire('user_data_fetch_scheduled');\r\n  }\r\n  \r\n  // Finally, require segments as usual.\r\n  segments = require('segments');\r\n}(Krux.require));\r\n</script>",
                "target": "",
                "target_action": null,
                "require": "",
                "docwrite": null,
                "method": "document",
                "execution_results": null,
                "tier": 2,
                "internal": 1,
                "content_type": "html",
                "timing": "asap",
                "type": "publisher",
                "id": 29295,
                "name": "Krux Load Segments v2"
            }, {
                "content": "<script>\r\n  Krux('require:http').pixel({\r\n    url: \"//p.brilig.com/contact/bct\",\r\n    data: {\r\n      pid: \"9354001b-d428-4e7c-9115-ffd042a2f184\",\r\n      _ct: \"pixel\",\r\n      puid: Krux('get', 'user'),\r\n      REDIR: \"http://beacon.krxd.net/data.gif?_kdpid=c3f63723-4104-46a9-9bda-e72960b2959a\"\r\n    }\r\n  });\r\n</script>",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "method": "document",
                "execution_results": null,
                "tier": 1,
                "internal": 20,
                "content_type": "html",
                "freq_cap": 3,
                "timing": "onload",
                "type": "data provider",
                "id": 37150,
                "name": "Brilig provider tag"
            }, {
                "content": "<script>\r\n(function(){\r\n  var kuid = Krux('get', 'user');\r\n  if (kuid) {\r\n    Krux('require:http').pixel({\r\n      url: \"//p.acxiom-online.com/pixel/sci\",\r\n      data: {\r\n          uid: kuid,\r\n          _kdpid: 'f19b7626-3732-4dcc-bac5-8d2c937dae9a',\r\n          pid: 3021\r\n      }});\r\n  }\r\n  })();\r\n</script>",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "method": "document",
                "execution_results": null,
                "tier": 1,
                "internal": 21,
                "content_type": "html",
                "freq_cap": 3,
                "timing": "onload",
                "type": "data provider",
                "id": 34354,
                "name": "Acxiom S2S provider tag"
            }, {
                "content": "<script>\r\n(function() {\r\n  // krxd.nexac.com throws SSL cert errors so only fire this tag\r\n  // on HTTP pages\r\n  if (location.protocol == \"http:\") {\r\n      var Krux = window.Krux;\r\n      var kuid = Krux('get', 'user');\r\n      if (kuid) {\r\n         var u = \"http://krxd.nexac.com/dlx.gif?_kdpid=2dd640a6-6ebd-4d4f-af30-af8baa441a0d&kuid=\" + kuid;\r\n         (new Image()).src = u;\r\n      }\r\n   }\r\n})();\r\n</script>",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "method": "document",
                "execution_results": null,
                "tier": 1,
                "internal": 8,
                "content_type": "html",
                "freq_cap": 3,
                "timing": "onload",
                "type": "data provider",
                "id": 34355,
                "name": "DataLogix provider tag"
            }, {
                "content": "<script>\r\n(function(){\r\n  if (window.KRUX) {\r\n    var pubid = KRUXSetup.pubid;\r\n  } else if (window.Krux) {\r\n    var pubid = Krux('get', 'pubid');\r\n  } else { \r\n    return;\r\n  }\r\n  var prefix = location.protocol == 'https:' ? \"https:\" : \"http:\";\r\n  var kurl = prefix + '//beacon.krxd.net/data.gif?_kdpid=890a6228-04af-4630-85b6-0b49dee6157f&pid=' + pubid;\r\n  var u = prefix + '//api.bizographics.com/v1/profile.redirect?api_key=595bae8dbc0c4c42b4544e688b10c002' + \r\n        '&callback_url=' + escape(kurl);\r\n  var i = new Image();\r\n  i.src = u;\r\n})();\r\n</script>",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "method": "document",
                "execution_results": null,
                "tier": 1,
                "internal": 2,
                "content_type": "html",
                "freq_cap": 3,
                "timing": "onload",
                "type": "data provider",
                "id": 34357,
                "name": "Bizo provider tag"
            }, {
                "content": "<script>\r\n(function(){\r\n  var kuid = Krux('get', 'user');\r\n  if (kuid) {\r\n    Krux('require:http').pixel({\r\n      url: \"//loadm.exelator.com/load\",\r\n      data: {\r\n          _kdpid: 'e4942ff0-4070-4896-a7ef-e6a5a30ce9f9',\r\n          buid: kuid,\r\n          p: '204',\r\n          g: '270',\r\n          j: '0'\r\n      }});\r\n  }\r\n  })();\r\n</script>",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "method": "document",
                "execution_results": null,
                "tier": 1,
                "internal": 11,
                "content_type": "html",
                "freq_cap": 3,
                "timing": "onload",
                "type": "data provider",
                "id": 34358,
                "name": "eXelate Media provider tag"
            }, {
                "content": "<script type=\"text/javascript\">Krux('social.init');</script>",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "internal": 1,
                "method": "document",
                "execution_results": null,
                "tier": 1,
                "library_tag_config": {},
                "content_type": "html",
                "timing": "onload",
                "type": "library",
                "id": 29292,
                "name": "Krux Track Social"
            }, {
                "content": "<script>\r\n// this tag is intentionally blank\r\n</script>",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "method": "document",
                "execution_results": null,
                "tier": 1,
                "internal": 6,
                "content_type": "html",
                "freq_cap": 3,
                "timing": "onload",
                "type": "data provider",
                "id": 29293,
                "name": "Technographic Data provider tag"
            }, {
                "content": "",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "method": "document",
                "execution_results": null,
                "tier": 1,
                "internal": 14,
                "content_type": "html",
                "freq_cap": 3,
                "timing": "onload",
                "id": 29294,
                "name": "Krux Geographic Data provider tag"
            }, {
                "content": "<script>\r\n// LiveRamp universal user match\r\n\r\n(function(){\r\n  var kuid = Krux('get', 'user');\r\n  if (kuid) {\r\n      var liveramp_url = 'https://idsync.rlcdn.com/379708.gif?partner_uid=' + kuid;\r\n      var i = new Image();\r\n      i.src = liveramp_url;      \r\n  }\r\n})();\r\n</script>",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "method": "document",
                "execution_results": null,
                "tier": 1,
                "internal": 63,
                "content_type": "html",
                "freq_cap": 3,
                "timing": "onload",
                "type": "data provider",
                "id": 40606,
                "name": "NetProspex provider tag"
            }, {
                "content": "<script>\r\n(function() {\r\n  // adadvisor redirects to http://logger... so it's not https safe\r\n  if (location.protocol == \"http:\") {\r\n     var u = \"https://adadvisor.net/adscores/g.js?sid=9212244187&_kdpid=2111c0af-fc3a-446f-ab07-63aa74fbde8e\";\r\n     (new Image()).src = u;\r\n   }\r\n})();\r\n</script>",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "method": "document",
                "execution_results": null,
                "tier": 1,
                "internal": 22,
                "content_type": "html",
                "freq_cap": 3,
                "timing": "onload",
                "type": "data provider",
                "id": 35500,
                "name": "AdAdvisor S2S provider tag"
            }, {
                "content": "<script>\r\n  Krux('require:http').pixel({\r\n    url: \"//s.ixiaa.com/digi/C726AB29-0470-440B-B8D2-D552CED3A3DC/a.gif\"\r\n  });\r\n</script>",
                "target": null,
                "target_action": "append",
                "require": "",
                "docwrite": null,
                "method": "document",
                "execution_results": null,
                "tier": 1,
                "internal": 28,
                "content_type": "html",
                "freq_cap": 3,
                "timing": "onload",
                "type": "data provider",
                "id": 35501,
                "name": "IXI Digital Open Market provider tag"
            }
            ],
            "segments": [],
            "publisher": {
                "uuid": "5d7aaa39-eeed-454f-bb7d-ecea38d770fc",
                "id": 12190,
                "name": "BBC"
            },
            "controltag_options": {
                "async": "true",
                "render": true
            },
            "site": {
                "id": 17080,
                "name": "BBC.com Homepage"
            },
            "dnt": null,
            "params": {
                "control_tag_load_sync": "false",
                "disable_http_compression": false,
                "remove_kxhead": true,
                "supertag_requires_approval": "true",
                "control_tag_namespace": null,
                "control_tag_stats_prefix": null,
                "beacon_host": "beacon.krxd.net",
                "control_tag_pixel_throttle": "100",
                "user_id_cookie": null,
                "max_segments": null,
                "no_pii": 0,
                "revenue_optimization": false,
                "context_terms": "false",
                "jslog_host": "jslog.krxd.net",
                "site_level_supertag_config": "site",
                "max_slot_time": 1000,
                "services_host": "apiservices.krxd.net",
                "capture_leakage": true,
                "client_side_storage": "localStorage,cookie",
                "userData_host": "cdn.krxd.net",
                "control_tag_version": "stable",
                "capture_js_errors": "true",
                "datatag_version": "3"
            },
            "services": {
                "impression": "//beacon.krxd.net/ad_impression.gif",
                "stats": "//apiservices.krxd.net/stats",
                "log": "//jslog.krxd.net/jslog.gif",
                "userData": "//cdn.krxd.net/user_data/segments/3",
                "optout": "//beacon.krxd.net/optout_check",
                "pixel": "//beacon.krxd.net/pixel.gif",
                "um": "//apiservices.krxd.net/um",
                "is_optout": "//beacon.krxd.net/optout_check",
                "set_optin": "//apiservices.krxd.net/consumer/optin",
                "social": "//beacon.krxd.net/social.gif",
                "set_optout": "//apiservices.krxd.net/consumer/optout",
                "data": "//beacon.krxd.net/data.gif",
                "event": "//beacon.krxd.net/event.gif"
            },
            "geo": {},
            "realtime_segments": [],
            "confid": "JA8mItOH"
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

