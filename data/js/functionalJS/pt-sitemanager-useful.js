
		$(document).ready(function() {
			checkForMaintenancePage();
		});

		function checkForMaintenancePage()
		{
			maintenancePageOn = "false";
			maintenanceMessage = "Popplet will be down from around 12-6am EST Thursday Sepetember 11 for maintenance.  Thank you for your patience.";
	
			console.log("maintenancePageOn: " + maintenancePageOn);
			console.log("maintenanceMessage: " + maintenanceMessage);					
			if(maintenancePageOn == "true") {
				$("body").html("<div id='maintenance'><img src='images/logo_200_blue.png' /><br />" + maintenanceMessage + "</div>");
				$("body").css("padding-top", "0px");
				$("body").css("background-color", "#8ed8f8");				
				$("body").css("line-height", "normal");				
				$("img").css("vertical-align", "baseline");
			}			
		}
    
    