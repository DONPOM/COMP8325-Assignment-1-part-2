
                 tacProp={vert:"Home",tags:"kvpagetype=homepage:atf:desktop;kvvert=premium;"};
                 tacProp.tags += ( (typeof HPAds!=="undefined") && (typeof HPAds.adtech_client_side_inf=="function") ) ? HPAds.adtech_client_side_inf() : "";
                 if (typeof adSetOthAT=="function") adSetOthAT(tacProp.tags);
              