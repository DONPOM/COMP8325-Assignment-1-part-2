
    Tumblr.assets_host       = 'https://secure.assets.tumblr.com';
        Tumblr.getRealNow = (function(p, ts) {
        var start = (p && p.timing && p.timing.responseStart) || Date.now();
        return function() { return ts + (Date.now() - start); };
    })(window.performance, 1416884786000);

    Tumblr.started_at = (new Date()).getTime();
    Tumblr.real_started_at = Tumblr.getRealNow();
            Tumblr._log_lady_bootstrap = {"keys":{"search_landing_usage":{"init_event":false,"keep_counts":true,"flags":[]},"search_session_v1":{"init_event":false,"keep_counts":false,"flags":[]}},"default_session":"llsid5473f233001094.46164442"};
    