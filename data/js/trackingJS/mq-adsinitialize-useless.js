
        if (m3.dotcom.MultiAuth) {
            m3.util.Event.add(window , "load" , function() {
                m3.dotcom.MultiAuth.init('https://api.screenname.aol.com', 'ao106HS28FhE50SM');
            });
        }

        m3.dotcom.controller.MCP.addSite('m3.dotcom.controller.Dotcom', {"cssProfiles":[["winston","screen, projection, print"],["winstonA","screen, projection, print"]],"enableP13N":true,"jsProfiles":["winston"],"type":"dotcom"});
        m3.dotcom.controller.MCP.boot('dotcom', {"model":{"active":null,"applications":[{"state":{"detailsId":null,"locations":null,"mapSearches":null,"mapState":null,"route":null},"type":"core"}],"businessLocator":null,"mapState":null,"type":"dotcom"},"defaultLocation":{"UUID":null,"address":{"adminArea":null,"approximate":false,"country":"AU","countryCode":null,"countryLong":"Australia","displayLatLng":null,"geocodeQualityCode":null,"latLng":{"lat":-33.867,"lng":151.207},"locality":"sydney","postalCode":null,"quality":"CITY","region":"","regionLong":null,"singleLineAddress":"sydney","slug":null,"street":null},"airport":false,"ambiguousResults":null,"bidToken":null,"branding":null,"breadcrumb":{},"bullets":null,"couponUrl":null,"csId":null,"displayOffset":null,"distance":null,"distanceFromUserHome":0,"errorId":null,"featBizCategory":null,"featBizCitySearchInfo":null,"gasPrices":null,"hotelHighPrice":null,"hotelLowPrice":null,"id":null,"idSearchLocation":false,"inputQuery":null,"legDistance":null,"menuUrl":"","message":null,"mostPopularCategory":null,"name":null,"notes":null,"numberOfRatings":null,"phone":null,"placesAtThisLocation":[],"placesPageURL":null,"position":0,"rating":null,"referenceId":null,"reservationUrl":"","routeDistance":null,"search":null,"seoOverrides":null,"sources":null,"status":"UNRESOLVED","teaser":"","thumbnail":null,"title":null,"toBeFormatted":false,"type":null,"unresolvedLocations":null,"visible":false,"website":"","yextTag":null,"yextUrl":null,"status":"UNRESOLVED","errorId":null}}); 

            // Initialize ads
m3.Ad.init();           
                $pv();
 
            m3.$("pageMask").style.display = "none";
            
    