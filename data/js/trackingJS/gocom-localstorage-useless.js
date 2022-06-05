

			$(function(){
				loadPageValueProp();

				// If third-party localstorage is disabled redirect to error page
				if (DisneyID.thirdPartyCookiesDisabled) {
					var page = "silent-client-functions",
						whiteList = ['cookies-disabled', 'silentlogin', 'silent-client-functions', 'recover-password', 'forgot-username-password', '500', '404', '403'];
					if ($.inArray(page, whiteList) === -1) {
						DisneyID.log("Error: third party cookies are disabled!");
						//redirect to error message
						DisneyID.navigate('cookies-disabled',{
							queryUpsert:{
								referer: document.location.href
							}
						});
						return;
					} else {
						DisneyID.log("Third party cookies are disabled, but page is whitelisted so things should be a-ok.",page);
					}
				}
			});

		
		if (DisneyID && typeof DisneyID.pageJS === 'function') {
			$(DisneyID.pageJS);
		}
		
		$(function() {
					});

			//to dynamically update tealium,
			//	utag.link({"form_name":"registration", flow_name":"fb", "field_name":'[lastName,firstName,...,terms]'})
		