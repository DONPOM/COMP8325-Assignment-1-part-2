FSR.surveydefs = [{
    site: 'adobe.com',
    name: 'browse',
    section: 'sitewide_ent',
    invite: {
        when: 'onentry'
    },
    pop: {
        when: 'later'
    },
    links: {
        pause: [{
            tag: 'a',
            attribute: 'href',
            patterns: ['/go/', '/buy/', 'cfusion/store', 'products']
        }
        ]
    },
    criteria: {
        sp: 1,
        lf: 1,

        locales: [{
            locale: 'fr',
            sp: 5,
            lf: 1
        }, {
            locale: 'de',
            sp: 5,
            lf: 1
        }, {
            locale: 'jp',
            sp: 1,
            lf: 1
        }, {
            locale: 'uk',
            sp: 5,
            lf: 1
        }
        ]
    },
    include: {
        urls: ['.']
    }
}
];
FSR.properties = {
    repeatdays: 90,

    repeatoverride: false,

    altcookie: {
        name: 'survey_cx',
        domain: 'adobe.com',
        path: '/',
        value: '1',
        persistent: true,
        repeatdays: 14
    },

    language: {
        locale: 'en_us',
        src: 'location',
        locales: [{
            match: '/fr/',
            locale: 'fr'
        }, {
            match: '/de/',
            locale: 'de'
        }, {
            match: '/uk/',
            locale: 'uk'
        }, {
            match: '/jp/',
            locale: 'jp'
        }
        ]
    },

    exclude: {},

    zIndexPopup: 10000,

    ignoreWindowTopCheck: false,

    ipexclude: 'fsr$ip',

    mobileHeartbeat: {
        delay: 60,
        max: 3600
    },

    invite: {
        siteLogo: "sitelogo.gif",
        siteLogoAlt: "",
        dialogs: [[{
            reverseButtons: false,
            headline: "",
            blurb: "Thank you for visiting Adobe's website. You have been selected to participate in a survey to let us know how we can improve adobe.com",
            noticeAboutSurvey: "Please help us by giving your feedback at the <u>end of your visit today</u>.",
            attribution: "This survey is conducted by ForeSee, an independent company, on behalf of Adobe Systems, Inc.",
            closeInviteButtonText: "Click to close.",
            declineButton: "No, thanks",
            acceptButton: "Yes, I'll help",
            locales: {
                "fr": {
                    headline: "",
                    blurb: "Merci de votre visite sur le site web d’Adobe.  Aidez nous à l’améliorer en faisant part de vos commentaires.",
                    noticeAboutSurvey: "",
                    attribution: "Cette enquête est menée par ForeSee, une société indépendante, travaillant pour le compte de Adobe Systems, Inc.",
                    closeInviteButtonText: "Cliquez pour fermer.",
                    declineButton: "Non merci",
                    acceptButton: "Oui, je souhaite participer"
                },
                "de": {
                    headline: "",
                    blurb: "Vielen Dank, dass Sie unsere Website besuchen!<br><br>Wir würden uns freuen, wenn Sie an einer Umfrage zur Kundenzufriedenheit teilnehmen und uns so dabei helfen, unsere Website weiter zu verbessern.",
                    noticeAboutSurvey: "Bitte geben Sie uns Ihr Feedback am Ende Ihres heutigen Besuchs auf unserer Website.",
                    attribution: "Diese Umfrage wird von einem unabhängigen Unternehmen, ForeSee, im Auftrag von Adobe Systems, Inc. durchgeführt.",
                    closeInviteButtonText: "Zum Schließen hier klicken",
                    declineButton: "Nein, danke",
                    acceptButton: "Ja, gerne"
                },
                "uk": {
                    headline: "",
                    blurb: "Thank you for visiting Adobe's website. You have been selected to participate in a survey to let us know how we can improve adobe.com",
                    noticeAboutSurvey: "Please help us by giving your feedback at the <u>end of your visit today</u>.",
                    attribution: "This survey is conducted by ForeSee, an independent company, on behalf of Adobe Systems, Inc.",
                    closeInviteButtonText: "Click to close.",
                    declineButton: "No, thanks",
                    acceptButton: "Yes, I will help"
                },
                "jp": {
                    headline: "",
                    blurb: "アドビのウェプサイトをご覧いただき、ありがとうございます。<br>お忙しいところ誠に恐れ入りますが、ウェブサイト改善のための 5 分程度の調査にご協力をお願いしてもよろしいですか。",
                    noticeAboutSurvey: "調査ページは、アドビのウェブサイトをご覧いただいた後に<u>自動的に</u>表示されます。",
                    attribution: "この調査はアドビシステムズの委託により、第三者調査機関である ForeSee Results 社が実施します。",
                    closeInviteButtonText: "この画面を閉じる",
                    declineButton: "辞退する",
                    acceptButton: "協力する"
                }
            }
        }
        ]],

        exclude: {
            urls: ['/cfusion/', '/newsletters/edge/', '/help/', '/ion/', '/brilliant/', 'labs.adobe.com', 'createpdf.adobe.com', 'partners.adobe.com/toolkit', 'adobemall.co.kr/AdobeMall/', '/software/flash/about/', 'photographersdirectory.adobe.com', 'partners.adobe.com/resellerfinder/na/findreseller.jsp', 'cooljobs.adobe.com/frameset.html', 'demo.adobe.com', 'sjw2.adobe.com/AdobeUserGroup/events.asp', 'sjw2.adobe.com/AdobeUserGroup/review.asp', 'onlineservices.adobe.com/account/tsu/a1', 'readstep2_servefile.html', 'readstep2_thankyou.html', 'readstep3_thankyou.html', 'macromedia.com/support/', '/jp/devnet/', '/de/devnet/', '/de/software/flash/about/', '/jp/software/flash/about/', 'service.stage.acrobat.com', 'service.acrobat.com', '/events/main.jsp', 'cps-internal.corp.adobe.com', 'omniture.com', /^(http|https):\/\/get(2?)\.adobe\.com/ , '/cart/checkout.html', '/fr/cart/checkout.html', '/de/cart/checkout.html', '/uk/cart/checkout.html', '/jp/cart/checkout.html'],
            referrers: ['adobe.com/downloads', /^http:\/\/get(2?)\./],
            userAgents: [],
            browsers: [],
            cookies: [],
            variables: []
        },
        include: {
            local: ['.']
        },

        delay: 0,
        timeout: 0,

        hideOnClick: false,

        hideCloseButton: false,

        css: 'foresee-dhtml.css',

        hide: [],

        hideFlash: false,

        type: 'dhtml',
        url: 'invite-mobile.html',
        back: 'url',
        SurveyMutex: 'survey_cx'
    },

    tracker: {
        width: '690',
        height: '415',
        timeout: 3,
        pause: 10,
        adjust: true,
        alert: {
            enabled: true,
            message: 'The survey is now available.',
            locales: [{
                locale: 'fr',
                message: 'Le questionnaire est désormais disponible.'
            }, {
                locale: 'de',
                message: 'Die Umfrage steht jetzt zur Verfügung.'
            }, {
                locale: 'uk',
                message: 'The survey is now available.'
            }, {
                locale: 'jp',
                message: 'ご回答をいただく準備が整いました。'
            }
            ]
        },
        url: 'tracker.html',
        locales: [{
            locale: 'fr',
            url: 'tracker_fr.html'
        }, {
            locale: 'de',
            url: 'tracker_de.html'
        }, {
            locale: 'uk',
            url: 'tracker_uk.html'
        }, {
            locale: 'jp',
            url: 'tracker_jp.html'
        }
        ]
    },

    survey: {
        width: 690,
        height: 600
    },

    qualifier: {
        footer: '<div div id=\"fsrcontainer\"><div style=\"float:left;width:80%;font-size:8pt;text-align:left;line-height:12px;\">This survey is conducted by an independent company ForeSee,<br>on behalf of the site you are visiting.</div><div style=\"float:right;font-size:8pt;\"><a target="_blank" title="Validate TRUSTe privacy certification" href="//privacy-policy.truste.com/click-with-confidence/ctv/en/www.foreseeresults.com/seal_m"><img border=\"0\" src=\"{%baseHref%}truste.png\" alt=\"Validate TRUSTe Privacy Certification\"></a></div></div>',
        width: '690',
        height: '500',
        bgcolor: '#333',
        opacity: 0.7,
        x: 'center',
        y: 'center',
        delay: 0,
        buttons: {
            accept: 'Continue'
        },
        hideOnClick: false,
        css: 'foresee-dhtml.css',
        url: 'qualifying.html'
    },

    cancel: {
        url: 'cancel.html',
        width: '690',
        height: '400'
    },

    pop: {
        what: 'survey',
        after: 'leaving-site',
        pu: false,
        tracker: true
    },

    meta: {
        referrer: true,
        terms: true,
        ref_url: true,
        url: true,
        url_params: false,
        user_agent: false,
        entry: false,
        entry_params: false
    },

    events: {
        enabled: true,
        id: true,
        codes: {
            purchase: 800,
            items: 801,
            dollars: 802,
            followup: 803,
            information: 804,
            content: 805
        },
        pd: 7,
        custom: {}
    },

    previous: false,

    analytics: {
        google_local: false,
        google_remote: false
    },

    cpps: {
        downloads: {
            source: 'url',
            init: 'None',
            patterns: [{
                regex: 'get.adobe.com',
                value: 'get'
            }, {
                regex: 'get2.adobe.com',
                value: 'get2'
            }
            ]
        }
    },

    mode: 'hybrid'
};

