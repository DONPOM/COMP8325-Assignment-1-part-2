
    var origURL = document.location.href;
    var truncatedURL = origURL.split("?", 1);
    var _comscore = _comscore || [];

    _comscore.push({ c1: "2", c2: "16955838", c4: truncatedURL[0]});

    (function()
    {
        var s = document.createElement("script"), el = document.getElementsByTagName("script")[0];

        s.async = true;
        s.src = (document.location.protocol == "https:" ? "https://sb" : "http://b") + ".scorecardresearch.com/beacon.js";
        el.parentNode.insertBefore(s, el);
    })();

