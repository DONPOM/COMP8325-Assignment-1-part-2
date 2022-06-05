var locales = {"africa": true, "at": true, "au": true, "be_en": true, "be_fr": true, "be_nl": true, "bg": true, "br": true, "ca": true, "ca_fr": true, "ch_de": true, "ch_fr": true, "ch_it": true, "cn": true, "cz": true, "de": true, "dk": true, "ee": true, "eeurope": true, "us": true, "es": true, "fi": true, "fr": true, "hk_en": true, "hk_zh": true, "hr": true, "hu": true, "ie": true, "il_en": true, "il_he": true, "in": true, "it": true, "kr": true, "la": true, "lt": true, "lu_de": true, "lu_en": true, "lu_fr": true, "lv": true, "mena_ar": true, "mena_en": true, "mena_fr": true, "mx": true, "nl": true, "no": true, "nz": true, "pl": true, "pt": true, "ro": true, "rs": true, "ru": true, "sk": true, "se": true, "sea": true, "si": true, "tr": true, "tw": true, "ua": true, "uk": true, "jp": true},
        tokens = document.location.pathname.split('/'),
        locale = ((tokens.length < 2) || ((locales[tokens[1]] == undefined) && tokens[1] != 'content')) ? 'us' : tokens[1],
        geoCook = 'international=',
        cookies = document.cookie.split(';');

    for (var i = 0; i < cookies.length; i++) {
        var c = cookies[i];
        if ((c.indexOf(geoCook) >= 0) && !(locale == 'content')) {
            // get the value of 'interntional='
            var preferredLocale = c.substring(c.indexOf(geoCook) + geoCook.length, c.length);

            if ((preferredLocale != locale) && (locales[preferredLocale])) {

                var url = document.location.pathname + document.location.search,
                    url = url.replace(/\/special\/dom\/tests\/homepage\/v10\/pages/,"") + window.location.search;

                if (locale == 'us') {
                    url = url.replace('/en/', '/');
                    url = '/' + preferredLocale + url;
                } else {
                    var suffix = url.substring(locale.length + 1);
                    if ((suffix == null) || (suffix == "")) {
                        suffix = "/";
                    }
                    if (preferredLocale != 'us') {
                        url = '/' + preferredLocale + suffix;
                    } else {
                        url = suffix;
                    }
                }

                window.location.replace(url);
            }
        }
    }