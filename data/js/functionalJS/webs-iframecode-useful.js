
	var webs = {};
	webs.locale = 'en-US';

	webs.props = {
		globalAssetBaseURL: 'http://static.websimages.com/v7800639',
		dynamicAssetServer: 'http://dynamic.websimages.com',
		premiumServer: 'https://premium.members.webs.com',
		secureServer: 'https://secure.members.webs.com',
		showSeo:true,
		environment: 'production',
		imageProcessorServer: 'http://imageprocessor.websimages.com',
		localSearchServer: 'http://locallistings.webs.prd.yyz.webs.com',
		membersServer: 'http://members.webs.com',
		showDomainTab: 'true'
	};
	webs.site = {
		id: parseInt('129132496', 10),
		url: 'http://imkhan.webs.com/',
		username: 'imkhan',
		verticalId: '1',
		creationDate: '2014-12-02 20:31:47.389',
		showCreateSiteLinks: true,
		showFreeImagesTab: true,
		logoutURL: '/s/logout',
		useNewStats: false,
		hasSeo:false,
		hasMobileLandingPage: false,
        hasMobilizedPages: true,
		enabledFeatures: ['EMAIL_VERIFICATION_EVENT','EMAIL_VERIFICATION_EMAIL_EPSILON','PREMIUM_WELCOME_EMAIL_EPSILON','EXPIRED_PREMIUM_BANNER','GLIMPSE','EXTERNAL_REDESIGN_2014','MODULE_PHOTO_GALLERY','THEME_FIRST_SIGNUP','PREMIUM_CHECKOUT_V3','PREMIUM_PURCHASE_EVENT','FORGOT_PW_EVENT','PREMIUM_PURCHASE_EMAIL','STATS','SITEBUILDER_CSS_ON_AKAMAI','MODULE_ASSETS_BURN_OUT','PREMIUM_WELCOME_SEND','DOMAIN_RENEWAL_EVENT_EPSILON','DOMAIN_RENEWAL_EVENT','WEBS_MOBILIZED_PAGES','WEBS_MOBILE','WELCOME_EVENT','WELCOME_EMAIL','EXPIRED_PREMIUM_EMAIL','NEW_PACKAGES_AND_PRICES','LOCAL_LISTINGS','MODULE_WEBSTORE_PRODUCTS','FORGOT_PW_EMAIL','PREMIUM_WELCOME_EMAIL','FORGOT_PW_EMAIL_EPSILON','PREMIUM_PURCHASE_EMAIL_EPSILON','WELCOME_EMAIL_EPSILON','TREE_DATA_BURN_OUT','DUNNING_RENEWAL_EVENT','DUNNING_RENEWAL_EMAIL','WEBS_EXPIRED_PREMIUM','EXTERNAL_LOCALE_SELECTOR'],
		noDomain: Boolean(false),
		hasPublished: Boolean(false),
		hasPremium: Boolean(false)
	};
	webs.permissions = {
		canChangePremium: Boolean(true)
	};
	webs.partner = {
		id: "295",
		cc: "US",
		isDefaultPartner: Boolean(false),
		vpHost: "",
		allowSocial : Boolean(true),
		showAppStoreLinks: true
	};

	webs.site.verticalName =
		webs.site.verticalId === 1 ? 'personal' :
		webs.site.verticalId === 2 ? 'group' :
		webs.site.verticalId === 3 ? 'business' : 'personal';
