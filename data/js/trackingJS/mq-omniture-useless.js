
        MQSTATICSERVER    = "http://web.mapquestapi.com";
        MQTRAFFSERVER     = "http://web.mapquestapi.com/traffic/v2";
        SEARCHMAPSERVER = "";
        TILEVERSION     = "";
        Key             = "Cmjtd%7Clu6t250ynq%2Cbl%3Do5-h625";
        IsDotCom        = true;
        MQSEARCHLAYERSERVERS = "tile1.searchlayer.mapquest.com,tile2.searchlayer.mapquest.com,tile3.searchlayer.mapquest.com,tile4.searchlayer.mapquest.com".split(',');
        SHOW_LEAD_PRICE = false;
        MQPROTOCOL = "http://";
        MQCDN = "http://content.mqcdn.com/winston-624/cdn/toolkit/lite/";
        MQICONSERVER = icons = "icons.mqcdn.com";
        MQPLATFORMSERVER = "http://web.mapquestapi.com";
        
        MQMAPCONFIG={"mapconfig":{"2":{"tilelogger":{"urlpattern":"http://coverage.tt.mqcdn.com/logger/v1/transaction?transaction=log&t={$type}&c={$cost1}&c2={$cost2}&s={$scale}&lat={$lat}&lng={$lng}&key={$key}&rand={$rand}","sslurlpattern":"https://coverage.tt.mqcdn.com/logger/v1/transaction?transaction=log&t={$type}&c={$cost1}&c2={$cost2}&s={$scale}&lat={$lat}&lng={$lng}&key={$key}&rand={$rand}"},"geofence":{"sslurl":"https://gpstrack.mapquest.com/tracking","url":"http://gpstrack.mapquest.com/tracking"},"geocode":{"sslurl":"https://web.mapquestapi.com/geocoding/","url":"http://web.mapquestapi.com/geocoding/"},"directions":{"sslurl":"https://web.mapquestapi.com/directions/","url":"http://web.mapquestapi.com/directions/"},"traffic":{"sslurl":"https://web.mapquestapi.com/traffic/","url":"http://web.mapquestapi.com/traffic/"},"staticmap":{"sslurl":"https://web.mapquestapi.com/staticmap/","url":"http://web.mapquestapi.com/staticmap/"},"cdn":{"sslurl":"https://api-s.mqcdn.com/","url":"http://api.mqcdn.com/"},"copyright":{"cat":["map","hyb","sat"],"zoom":{"hi":"18","low":"0"},"urlpattern":"http://coverage.tt.mqcdn.com/coverage?projection=sm&format={$format}&loc={$lon1},{$lat1},{$lon2},{lat2}&zoom={$zoom}&cat={$category}","format":["xml","json"],"sslurlpattern":"https://coverage.tt.mqcdn.com/coverage?projection=sm&format={$format}&loc={$lon1},{$lat1},{$lon2},{lat2}&zoom={$zoom}&cat={$category}"},"maplayer":{"vmap":{"description":"vector geometry for map","urlpattern":"http://ttiles0{$hostrange}.mqcdn.com/tiles/1.0.0/vy/vmap/{$z}/{$x}/{$y}.{$ext}","ext":["pbf"],"sslurlpattern":"https://ttiles0{$hostrange}-s.mqcdn.com/tiles/1.0.0/vy/vmap/{$z}/{$x}/{$y}.{$ext}","hostrange":{"lo":"1","hi":"4"}},"tflow":{"description":"alternate vendor traffic flow","urlpattern":"http://ttiles0{$hostrange}.mqcdn.com/tiles/1.0.0/vy/tflow/{$z}/{$x}/{$y}.{$ext}","ext":["png"],"sslurlpattern":"https://ttiles0{$hostrange}-s.mqcdn.com/tiles/1.0.0/vy/tflow/{$z}/{$x}/{$y}.{$ext}","hostrange":{"lo":"1","hi":"4"}},"mobmap":{"description":"alternate vendor base map layer sans terrain","urlpattern":"http://ttiles0{$hostrange}.mqcdn.com/tiles/1.0.0/vy/mobmap/{$z}/{$x}/{$y}.{$ext}","ext":["jpg"],"sslurlpattern":"https://ttiles0{$hostrange}-s.mqcdn.com/tiles/1.0.0/vy/mobmap/{$z}/{$x}/{$y}.{$ext}","hostrange":{"lo":"1","hi":"4"}},"hyb":{"description":"alternate vendor transparent hybrid overlay layer","urlpattern":"http://ttiles0{$hostrange}.mqcdn.com/tiles/1.0.0/vy/hyb/{$z}/{$x}/{$y}.{$ext}","ext":["png"],"sslurlpattern":"https://ttiles0{$hostrange}-s.mqcdn.com/tiles/1.0.0/vy/hyb/{$z}/{$x}/{$y}.{$ext}","hostrange":{"lo":"1","hi":"4"}},"sat":{"description":"alternate vendor aerial imagery layer","urlpattern":"http://ttiles0{$hostrange}.mqcdn.com/tiles/1.0.0/vy/sat/{$z}/{$x}/{$y}.{$ext}","ext":["jpg"],"sslurlpattern":"https://ttiles0{$hostrange}-s.mqcdn.com/tiles/1.0.0/vy/sat/{$z}/{$x}/{$y}.{$ext}","hostrange":{"lo":"1","hi":"4"}},"tgeom":{"description":"alternate vendor road geometry for traffic","urlpattern":"http://ttiles0{$hostrange}.mqcdn.com/tiles/1.0.0/vy/tgeom/{$z}/{$x}/{$y}.{$ext}","ext":["json"],"sslurlpattern":"https://ttiles0{$hostrange}-s.mqcdn.com/tiles/1.0.0/vy/tgeom/{$z}/{$x}/{$y}.{$ext}","hostrange":{"lo":"1","hi":"4"}},"map":{"description":"alternate vendor base map layer","urlpattern":"http://ttiles0{$hostrange}.mqcdn.com/tiles/1.0.0/vy/map/{$z}/{$x}/{$y}.{$ext}","ext":["jpg"],"sslurlpattern":"https://ttiles0{$hostrange}-s.mqcdn.com/tiles/1.0.0/vy/map/{$z}/{$x}/{$y}.{$ext}","hostrange":{"lo":"1","hi":"4"}},"vsathyb":{"description":"vector geometry for hyb and sat raster","urlpattern":"http://ttiles0{$hostrange}.mqcdn.com/tiles/1.0.0/vy/vsathyb/{$z}/{$x}/{$y}.{$ext}","ext":["pbf"],"sslurlpattern":"https://ttiles0{$hostrange}-s.mqcdn.com/tiles/1.0.0/vy/vsathyb/{$z}/{$x}/{$y}.{$ext}","hostrange":{"lo":"1","hi":"4"}},"tpattern":{"description":"alternate vendor traffic patterns","urlpattern":"http://ttiles0{$hostrange}.mqcdn.com/tiles/1.0.0/vy/tpattern/{$z}/{$x}/{$y}.{$ext}","ext":["json"],"sslurlpattern":"https://ttiles0{$hostrange}-s.mqcdn.com/tiles/1.0.0/vy/tpattern/{$z}/{$x}/{$y}.{$ext}","hostrange":{"lo":"1","hi":"4"}},"sathyb":{"description":"alternate vendor hybrid overlayed on aerial imagery","urlpattern":"http://ttiles0{$hostrange}.mqcdn.com/tiles/1.0.0/vy/sathyb/{$z}/{$x}/{$y}.{$ext}","ext":["jpg"],"sslurlpattern":"https://ttiles0{$hostrange}-s.mqcdn.com/tiles/1.0.0/vy/sathyb/{$z}/{$x}/{$y}.{$ext}","hostrange":{"lo":"1","hi":"4"}}}},"info":{"copyright":{"text":"© 2014 MapQuest, Inc.","imageUrl":"http://api.mqcdn.com/res/mqlogo.gif","imageAltText":"© 2014 MapQuest, Inc."},"statuscode":0,"messages":[]},"version":4}};
        
        SITECONFIG = {
            otaScriptURL:'http://content.mqcdn.com/ota-cdn-release-17.0-85/ota.production.min.js',
            mapServer:'mq-devhost-lm21.ihost.aol.com',
            mapPort: 8600,
            mapPath: 'mq',
            cdnPath: 'http://content.mqcdn.com/winston-624',
            aolImgResize: 'http://o.aolcdn.com/dims-global/dims3/MAPQUEST/',
            secureBase: 'https://www.mapquest.com/',
            tinyUrlBase: 'http://mapq.st',
            neighborhoodVibeUrl: 'http://local.mapquest.com',
            neighborhoodVibeAPIUrl: 'http://mqvibe-api.mapquest.com/places/search/',
            localUrl: 'http://local-stage.mapquest.com/',
            enableClassicInterstitial: 'true' == 'true',
            enableSTO: 'true' == 'true',
            enableSync: 'true' == 'true',
            enableGarmin: 'true' == 'true',
            displayDisabledFacebookMsg: 'false' == 'true',
            displayDisabledOnStarMsg: 'false' == 'true',
            enableFacebook: 'true' == 'true',
            syncUrl:  'https://xgtws.ford.com/5001/SendToSync',
            stoUrl:  'https://www.onstar.com/web/portal/odm',
            searchMapTimeout: 15000,
            searchMapAlwaysAlongRoute: true,
            maxStops: 26,
            enableTypeAhead: true,
            typeAheadMinScore: 0.91,
            typeAheadKeyLimit: 1,
            enableSearchLayer: true,
            enableTransit: 'true' == 'true',
            enablePedestrian: 'true' == 'true',
            enableBicycle: 'true' == 'true',
            reuseSearchTab: false,
            minimizeInsetMap: true,
            facebookStaticMapKey: 'Kmjtd|luub290anq%2C8w%3Do5-lzznh',
            facebookStaticMapServer: 'http://web.mapquestapi.com/staticmap/v4/getmap',
            adCopyKey: 'YFuMaeQ1NaioDJ8RRx4pQa4xgiu1a2UK',
            adCopyUrl: 'http://api.solvemedia.com/papi/challenge.ajax',
            enableMouseWheelZoom: true,
            altRouteMax: '3',
            isOSM: false,
            osmUseNominatim: false,
            profileCookieDomain: 'mapquest.com',
            enableDrag: true,
            browserPreview: false,
            tacodaUrl:  '/cdn/html/tacoda.html',
            tacodaCatBrandUrl:  '/cdn/html/tacoda_catbrand.html',
            citysbestDetailsUrl: 'http://www.citysbest.com/mapquest/',
            citysbestDetailsEnable: false,
            revision:'winston.null',
            enableReloadDialog: true,
            enableSearchFeedback: 'true' == 'true',
            enableFiveBoxForm: false,
            yextMax: 100,
            yextId: '476af49678',
            enableGeolocation: true,            
            enableVendorSourceDebug: 'false' == 'true',
            enableFormatDynamics: false,
            detailPagesCMSUrl: 'http://dataingestion-ntc.iweb.mapquest.com/cms',
            referringUrl: '',
            uacServerOverride: '',
            uacNetIdOverride: '',
            isDisplayAdIntl: '',
            displayAdServerOverride: '',
            displayAdNetIdOverride: '',
            yextWhitelabelDetailsEnable: 'true' == 'true',
            yextWhitelabelDetailsTemplateID: '54',
            yextPortalUrl: 'http://www.yext.com/pl/mapquest-claims/',
            infoGroupDetailsTemplateID: '9',
            yextAddPhotosLinkEnable: 'true' == 'true',
            yextOwnerVerifiedLinkEnable: 'true' == 'true',
            claimMyBusinessLinkEnable: 'true' == 'true',
            infogroupAddPhotosLinkEnable: 'true' == 'true',
            enableGasPrices: true,
            drivingRouteColor: '',
            enableRoundTrip: true,
            enableNeighborhoods: 'false' == 'true',
            dataLayerEnabled: true,
            yextPowerListingEnable: 'true' == 'true',
            yextPowerListingTemplateID: '54',
            trafficDisableCameras: 'false' == 'true',
            welcomePromoMode: 'interstitial',
            welcomeInterstitialType: '',
            roadshieldUrl:'icons.mqcdn.com/icons/roadsign.gif', 
            hotelsCheckPriceLinkEnable: 'true' == 'true',
            hotelsToolTip: 'Book Now with MapQuest', 
            patchEnabled: 'false' == 'true', 
            patchStoriesEnabled: 'false' == 'true',
            enableHideSteps: 'true' == 'true',
            enableVehiclesAndFuelCosts: 'true' == 'true',
            irsReimbursementRate: '0.56',
            enableHideSteps: 'true' == 'true',
            disableUACAds: '',
            disableHTMLAds: '',
            disableYaacMinHeightRequirement:  'false' == 'true',
            omniTrackingServer: 'o.sa.aol.com',
            omniTrackingServerSecure: 's.sa.aol.com',
            placePagesBaseUrl: 'http://www.mapquest.com',
            enableTacoda: 'true',
            enableReversePoiLookup: 'true' == 'true',
            placesAtThisAddressMaxResults: '10',
            otaPromoPages: 'MapResults,DirectionsResults,BizLocator,SearchResults-Inline,SearchOnMap,SearchNearby,SearchAlongRoute',
            yaacPromoPages: '',
            yaacNoPromoBrowsers: 'ie7',
            yaacSingleCollectionsOn: 'false',
            yaacBoundingBoxMargin: '150,150',
            enableOmniturePageViewTracking: 'true' == 'true',            
            enableOmnitureActionTracking: 'false' == 'true',
            enableGoogleAnalyticsActionTracking: 'true' == 'true',
            farm:  'afarm' ,
            pricelineRefid: '5101',
            priceSourceVendorIDs: '66',
            enableDirectionsPrintSendEnable: 'true' == 'true',
            enableDirectionsAutoScroll: 'true' == 'true',
            getFlyinAdLocationUsingDirectionsInputs: 'true' == 'true',
            testbed:  '' ,
            ypSearchImpressionURL: 'http://c.ypcdn.com/2/s/rtd',
            ypClickTrackingURL: 'http://c.ypcdn.com/2/c/rtd',
            ypPartnerId: 'l6wqdpvb9k',
            cmsBatchLimit: 26,
            enableMOAT: 'true' == 'true',
            enableAdSetAddOn: 'false' == 'true',
            send2CellAppID: '11',
            send2CellAppKey: 'ao1kb7Gw0lAN_9_f',
            enableMobileAppPromos: 'true' === 'true',
            enableHomePageLeftPanePromo: 'true' === 'true',
            korrelateBaseUrl: 'http://a02.korrelate.net/a/e/d2a.ads?et=a&ago=127&ao=128&px=123',
            korrelateEnabled: 'false' === 'true',
            accountsBaseUrl: 'https://accounts.mapquest.com',
            accountsClientID: '47559b9530487678a54baf4746a5506f541fba9c7815aec4f5000869a49d3a8b',
            accountsCallbackUrl: 'https://www.mapquest.com/_svc/auth/signin',
            mobileAppPromoAltBg: 'mobile_app_promo_main_mlb.jpg ',
            mobileAppPromoAltConfBg: 'mobile_app_promo_confirmation_mlb.jpg',
            enableAolToolbar: 'true' === 'true',
            enableAolToolbarAlways: 'false' === 'true',
            aolToolbarId: 'ao106HS28FhE50SM',
            aolToolbarUrl: '//o.aolcdn.com/os/aol/unb.min.js',
            lomTrafficPercentage: '100',
            enableLeaveBehind: 'true' === 'true',
            send2EmailFrom: 'no-reply@mapquest.com',
            enableOriginCategoryBrand: 'true'
        };

        PROMO = {
        	"max-stops-message": { eDate: '', text: 'We&#39;re sorry. You&#39;ve reached the current max number for stops.'},
        	"right-header-ext": { eDate: '2015-09-01 00:00:00.0', text: '<div id="twitterFollow">            <iframe id="iframeTwitter" scrolling="no" frameborder="0" style="border: medium none; overflow: hidden; height: 21px;" src="http://platform.twitter.com/widgets/follow_button.html?screen_name=mapquest&show_screen_name=false" allowtransparency="true"></iframe></div><div id="facebookLike"> <iframe src="http://www.facebook.com/plugins/like.php?href=http://www.facebook.com%2FMapQuest&amp;send=false&amp;layout=button_count&amp;width=90&amp;show_faces=true&amp;locale=en_US&amp;action=like&amp;colorscheme=light&amp;font=verdana&amp;height=21" scrolling="no" frameborder="0" style="border:none; overflow:hidden; width:90px; height:21px;" allowTransparency="true"></iframe>        </div><style type="text/css">#twitterFollow #iframeTwitter {  width: 60px;}.ie7 #twitterFollow #iframeTwitter {  width: 60px;}</style>'},
        	"gasprices": { eDate: '', text: ''},
        	"traffic-toggle-sponsorship": { eDate: '', text: ''},
        	"traffic-sponsorship": { eDate: '', text: ''},
        	"traffic-camera-sponsorship": { eDate: '', text: ''},
        	"hotels-tracking": { eDate: '', text: '{    "maps_price": {        "prefix": "http://ad.doubleclick.net/clk;252264604;76336915;k",        "sufix" : "rffrid=mdp.hcom.us.185.008.02.oimapsprice"    },    "maps_reservation": {        "prefix": "http://ad.doubleclick.net/clk;252264604;76336918;n",        "sufix": "rffrid=mdp.hcom.us.185.008.02.oimapsreservation"    },    "infowindow_price": {        "prefix": "http://ad.doubleclick.net/clk;252264604;76336924;k",        "sufix": "rffrid=mdp.hcom.us.185.008.02.oiinfowndprice"    },    "infowindow_reservation": {        "prefix": "http://ad.doubleclick.net/clk;252264604;76336932;j",        "sufix": "rffrid=mdp.hcom.us.185.008.02.oiinfowndreservation"    },    "search_price": {        "prefix": "http://ad.doubleclick.net/clk;252264604;76336893;p",        "sufix": "rffrid=mdp.hcom.us.185.008.02.oisearchprice"    },    "search_reservation": {        "prefix": "http://ad.doubleclick.net/clk;252264604;76336907;l",        "sufix": "rffrid=mdp.hcom.us.185.008.02.oisearchreservation"    },    "directions_price": {        "prefix": "http://ad.doubleclick.net/clk;252264604;76336940;i",        "sufix": "rffrid=mdp.hcom.us.185.008.02.oidirsrsltprice"    },    "directions_reservation": {        "prefix": "http://ad.doubleclick.net/clk;252264604;76336941;j",        "sufix": "rffrid=mdp.hcom.us.185.008.02.oidirsrsltreservation"    },    "bizloc_price": {        "prefix": "http://ad.doubleclick.net/clk;252264604;76336942;k",        "sufix": "rffrid=mdp.hcom.us.185.008.02.oihtlbizlocprice"    },    "bizloc_reservation": {        "prefix": "http://ad.doubleclick.net/clk;252264604;76336943;l",        "sufix": "rffrid=mdp.hcom.us.185.008.02.oihtlbizlocreservation"    }   }'},
            "hotels-reservation-text": { eDate: '', text: 'Check Availability'},
        	"roadblock-promo-template":  '' 
        };
        
        
        ROADBLOCK_WIDGET_RESULTS = {
          "results":{}, "region_name":"Sydney"
		};

<!-- static promo ClaimThisBiz-ExcludedVendors -->
SITECONFIG.excludedSourceVendors='34005,34315,34566,34568,34579,34580,34647,34648,34651,34652,34655,34678,36709,36950,36966,36981,36982,36983,36992,37039,37052,37053,37064,37065,37066,37074,34610,34626,34646,34649,34650,34653,34654,34656,34671,34672,36757,36838,36993,37015,25,54,37133,37113,37158,66';

        DOMAIN = {"country":"US","defaultLatLng":null,"distanceUnit":"m","domain":"com","facebookKey":null,"features":null,"garminKey":null,"garminSite":null,"locale":"en_US","omnitureProps":null,"omnitureSuite":"aolsvc","omnitureSuiteMQ":null,"sendToFeatures":null};
        USER = {"anonymous":true,"birthDate":null,"contacts":null,"devices":null,"displayName":null,"email":null,"firstName":null,"firstUpgrade":false,"hideStepsIds":null,"history":null,"id":null,"identities":null,"languagesTabDone":false,"lastName":null,"modifyDate":null,"myedits":null,"settings":{"defaultCountry":null,"defaultLocation":null,"distanceUnit":null,"insetMapPreference":"max","language":null,"locale":null,"temperatureUnit":null},"startaddress":null,"status":null,"type":"ANON","unauthIds":null,"uncategorized":null,"upgradeDate":null,"upgradeStatus":null,"vehicles":null,"welcomeTabDone":false};
            USER_HOME = {"lat":-33.867,"lng":151.207};
        
        SUPPRESS_ANIMATION = false;
<!-- static promo dma-map -->
DMA_MAP = {
    'NEWYORK': {locality:'New York', region:'NY', hpVertical:'NY'},
    'DENVER': {locality:'Denver', region:'CO', hpVertical:'Denver'},
    'CHICAGO': {locality:'Chicago', region:'IL', hpVertical:'Chicago'},
    'LOSANGELES': {locality:'Los Angeles', region:'CA', hpVertical:'LA'},
    'SANFRANCISCO': {locality:'San Francisco', region:'CA', hpVertical:'SF'}
}
        
        //TODO - FB keys for intl sites - different key per domains
        if (!DOMAIN || !DOMAIN.country || DOMAIN.country.toUpperCase() == "US" || !true || !DOMAIN.facebookKey){
            SITECONFIG.facebookKey = '324090988325';
        } else {
            SITECONFIG.facebookKey = DOMAIN.facebookKey;
        }
        
        SKIN_EVENT_HOLDER = null;

        // ------------------------------------------------------------
        // Stubbed Omniture stuff in case omniunih.js is blocked.
        var s_265 = {sa: function(){},getQueryParam: function(){}},
            s_gi = function(){},
            htmlAdWH = null,
            adSetOthAT = null;
        // ------------------------------------------------------------

    