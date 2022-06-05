
   "use strict";
    
    function _adobe_getProducts () {
        "use strict";
        
        var pathname = window.location.pathname,
            products = '',
            categoryPortion = '',
            productPortion = '',
            pathnameSplit = [],
            countryUrls = {
                'africa': 1,
                'at': 1,
                'au': 1,
                'be_en': 1,
                'be_fr': 1,
                'be_nl': 1,
                'bg': 1,
                'br': 1,
                'ca': 1,
                'ca_fr': 1,
                'ch_de': 1,
                'ch_fr': 1,
                'ch_it': 1,
                'cin': 1,
                'cn': 1,
                'cz': 1,
                'de': 1,
                'dk': 1,
                'ee': 1,
                'eeurope': 1,
                'es': 1,
                'fi': 1,
                'fr': 1,
                'hk_en': 1,
                'hk_zh': 1,
                'hr': 1,
                'hu': 1,
                'ie': 1,
                'il_en': 1,
                'il_he': 1,
                'in': 1,
                'it': 1,
                'jp': 1,
                'kr': 1,
                'la': 1,
                'lt': 1,
                'lu_de': 1,
                'lu_en': 1,
                'lu_fr': 1,
                'lv': 1,
                'mena_ar': 1,
                'mena_en': 1,
                'mena_fr': 1,
                'mx': 1,
                'nl': 1,
                'no': 1,
                'nz': 1,
                'pl': 1,
                'pt': 1,
                'ro': 1,
                'rs': 1,
                'ru': 1,
                'se': 1,
                'sea': 1,
                'si': 1,
                'sk': 1,
                'tr': 1,
                'tw': 1,
                'ua': 1,
                'uk': 1,
                'us': 1
            };
        
        //------------------------------------------------------------------------------
        // Pathname
        //------------------------------------------------------------------------------
        // remove language_locale from path
        // we do it this way to ensure that it only occurs in the 
        // beginning of the string
        pathnameSplit = pathname.split('/');
        if (pathnameSplit.length >= 2 && countryUrls[pathnameSplit[1]]) {
            pathnameSplit.splice(1, 1);
        }
        
        if (pathnameSplit.length >= 2) {
            categoryPortion = pathnameSplit[1].toLowerCase();
            // adobe.com/products/*
            if (categoryPortion === 'products') {
                if (pathnameSplit.length >= 3) {
                    productPortion = pathnameSplit[2].toLowerCase();
                    // photoshop-lightroom first so that we don't accidentally 
                    // assume photoshop
                    if (productPortion.indexOf('photoshop-lightroom') !== -1) {
                        products = 'PhotoshopLightroom';
                    } else if (productPortion.indexOf('photoshop') !== -1) {
                        products = 'Photoshop';
                    } else if (productPortion.indexOf('illustrator') !== -1) {
                        products = 'Illustrator';
                    } else if (productPortion.indexOf('indesign') !== -1) {
                        products = 'InDesign';
                    } else if (productPortion.indexOf('premiere') !== -1) {
                        products = 'PremierePro';
                    } else if (productPortion.indexOf('aftereffects') !== -1) {
                        products = 'AfterEffects';
                    } else if (productPortion.indexOf('dreamweaver') !== -1) {
                        products = 'Dreamweaver';
                    } else if (productPortion.indexOf('muse') !== -1) {
                        products = 'Muse';
                    } else if (productPortion.indexOf('flash') !== -1) {
                        products = 'FlashP';
                    }
                }
            // adobe.com/creativecloud.html
            } else if (categoryPortion.indexOf('creativecloud') !== -1) {
                products = 'CreativeCloud';
            }
        }
        
        return products;
    }
