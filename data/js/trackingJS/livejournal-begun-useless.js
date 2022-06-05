
    (new Image).src = [
        "https:" === document.location.protocol ? "https:" : "http:",
        "//autocontext.begun.ru/analytics?target_id=0&counter_id=0&url=",
        encodeURIComponent(document.URL),
        "&ref=", encodeURIComponent(document.referrer),
        "&rnd=", Math.random()
    ].join('');