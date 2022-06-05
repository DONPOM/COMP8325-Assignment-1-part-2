
     
     var doneBindingHospital = false;
     var doneBindingLanguage = false;
     
   


     $(document).ready(function() {

         $.support.cors = true;

              
        
         //allow only numeric characters on zipcode field
         $("#txtZip").keypress(function(event) {
             var code = event.charCode || event.keyCode;
             if (event.keyCode == 8)
                 return true;
             else {
                 return /\d/.test(String.fromCharCode(code));
             }

         });


         // get data for autocomplete controls...
         setAutoCompleteControls();

         // get data for dropdown controls...
         setDropDowns();

       

         //on zip code->change event deactivate or activate the distance and hospital choices
         $("#txtZip").change(function() {
             var zip = $("#txtZip").val();

             if (zip.length > 0) {
                 $("#selHospital").attr("disabled", "disabled");
             }
             else {
                 $("#selHospital").attr("disabled", false);
             }

         });

         $("#selHospital").change(function() {
             var hosp = $("#selHospital").val();

             if (hosp != 0) {
                 $("#txtZip").attr("disabled", "disabled");
                 $("#selDistance").attr("disabled", "disabled");
             }
             else { //Any Hospital == 0

                 $("#txtZip").attr("disabled", false);
                 $("#selDistance").attr("disabled", false);
                 $("#selDistance").val(0); //on reenable, reset it to default value too.
             }

         });


         $("#txtLName").keydown(function(event) {
             if (event.keyCode == 13) {

                 runNameSearch();
             }
         });
         $("#txtKeyword").keydown(function(event) {
             if (event.keyCode == 13) {
                 runSearch();
             }
         });

         // click handlers for the search buttons...
         $("#cmdSearch1").click(function() { runNameSearch(); });
         $("#cmdSearch2").click(function() { runSearch(); });

         $('#resetbutton').bind('click', function() { resetAll(); });

     });
                
                /* 
                Check if the query parmaters exists and select the fields according           
                */
                
                function initFromQS() {

                 var qs = new QueryStringParser();

                 var keyword = qs.getValue("keyword");
                 var zip = qs.getValue("ZIP");

                 if (keyword != "") {
                     $("#txtKeyword").val(keyword);
                 }
                 if (zip != "") {
                     hasQueryParameters = true;
                     $("#txtZip").val(zip);
                 }

                 var lang = qs.getValue("language");
                 if (lang.trim() != "") {
                     var id = "#selLanguage option[value=" + lang.toUpperCase() + "]";
                     $(id).prop("selected", true);
                 }

                 var hosp = qs.getValue("hospital");
                 if (hosp.trim() != "") {
                     //hosp = hosp.replace("’", "\\'"); //may not need this as hospital code will be passed in
                     var id = "#selHospital option[value=" + hosp + "]";
                     $(id).prop("selected", true);

                 }
             }

             /*
            Check if the all the asyn bindings of dropdowns is complete before checking and setting the query paramters
            */
             function checkQStrings() 
             {
                 if(doneBindingHospital && doneBindingLanguage)
                    initFromQS();
             }

            /*
             * Requests values for the form's drop-down boxes.
             */
            function setDropDowns() {
              

                // populate the hospital drop-down...
                $.ajax({
                    type: "GET",
                    url: endpointAddr + "/GetActiveFacilities",
                    dataType: "jsonp",
                    contentType: "application/json",
                    success: function(data) {
                        $("#selHospital").append($("<option />").val(0).text("Any Hospital"));
                        $.each(data.GetActiveFacilitiesResult, function() {
                            $("#selHospital").append($("<option />").val(this.cc_fac_no).text(this.FacilityName));
                        });
                        doneBindingHospital = true;
                        checkQStrings();//binding is done, now check the query paramters
                    }
                });


                // populate the language drop-down...
                $.ajax({
                    type: "GET",
                    url: endpointAddr + "/GetLanguages",
                    dataType: "jsonp",
                    contentType: "application/json",
                    success: function(data) {
                        $("#selLanguage").append($("<option />").val(0).text("Language Spoken"));
                        $("#selLanguage").append($("<option />").val(1).text("English"));
                        $("#selLanguage").append($("<option />").val(2).text("All Languages"));
                        $.each(data.GetLanguagesResult, function() {
                            $("#selLanguage").append($("<option />").val(this.LanguageCode).text(this.Name));
                        });
                        doneBindingLanguage = true;
                        checkQStrings();//binding is done, now check the query paramters
                    }
                });
            }


            /*
             * Creates the jQuery UI autocomplete functionality for textbox controls.
             */
            function setAutoCompleteControls() {
              
                // keyword: search specialties
                $("#txtKeyword").autocomplete({
                    minLength: 2,
                    source: function (request, response) {
                        $.getJSON(endpointAddr + "/GetSpecialtyList/" + request.term+ "?callback=?", request,
                        function (data) {
                           
                            response(data.GetSpecialtyListResult);
                        });
                    }
                });


                // insurance: search carriers
                $("#txtInsurance").autocomplete({
                    minLength: 2,
                    source: function (request, response) {
                        $.getJSON(endpointAddr + "/GetInsurance/" + request.term+ "?callback=?", request,
                        function (data) {
                            response(data.GetInsuranceResult);
                        });
                    }
                });

              
                // zip code: match by code
                $("#txtZip").autocomplete({
                    minLength: 2,
                    source: function(request, response) {
                        $.getJSON(endpointAddr + "/GetZipCode/" + request.term + "?callback=?", request,
                                function(data) {
                                    response(data.GetZipCodeResult);
                                });
                    }
                });

                //physician name: search by last name
                $("#txtLName").autocomplete({
                    minLength: 2,
                    source: function (request, response) {
                    $.getJSON(endpointAddr + "/GetPhysicianNames/" + request.term + "?callback=?", request,
                        function (data) {
                            response(data.GetPhysicianNamesResult);
                        });
                    }
                });
            }
      
         

            // ------------------------------------------------------------------------------------------
            // can't find these in the markup we were given; this should be more or less equivalent...
            function clearMe(obj) {
                if (obj != null && obj.value == obj.defaultValue) {
                    obj.value = "";
                }
            }

            function resetAll() {
                $("#txtKeyword").val("Condition, Disease or Physician Specialty...");
                $("#txtLName").val("");
                $("#txtZip").val("ZIP Code");
                $("#selGender").val(0);
                $("#selHospital").val(0);
                $("#selLanguage").val(0);
                $("#selDistance").val(0);
                $("#txtInsurance").val("Insurance Carrier");
                $("#selHospital").attr("disabled", false);
                $("#txtZip").attr("disabled", false);
                $("#selDistance").attr("disabled", false);
            }
            function clickrecall(obj, str) {

                if (obj.value == "") {
                    obj.value = obj.defaultValue;
                }
               
            }

            // ...no idea about these, though...
            function toggleStatus() { }
            function toggleStatus2() {
                    //$("#selDistance").attr("disabled", $.trim($("#txtZip").val()) != "" ? "" : "disabled");


             }

            // ------------------------------------------------------------------------------------------

            function runSearch() {
               
                var zip = $("#txtZip").val();
                var dist = $("#selDistance").val();
                var kwd = $("#txtKeyword").val();
                var gen = $("#selGender").val();
                var lang = $("#selLanguage").val();
                var ins = $("#txtInsurance").val();
                var hosp = $("#selHospital").val();

               

                var queryString = searchResultsURL;
                if (kwd.length > 0 && kwd != "Condition, Disease or Physician Specialty...") {
                    queryString = queryString + "?keyword=" + encodeURIComponent(kwd);
                   
                }
                if (hosp != 0) {
                    if (queryString.indexOf("?") == -1) {
                        queryString = queryString + "?hospital=" + encodeURIComponent(hosp);
                    }
                    else
                        queryString = queryString + "&hospital=" + encodeURIComponent(hosp);
                                         
                }
                if (zip.length == 5 && $.isNumeric(zip) && zip != "ZIP Code") {
                    if(kwd != "Condition, Disease or Physician Specialty...")
                        queryString = queryString + "&location=" + encodeURIComponent(zip) + "&distance=" + encodeURIComponent(dist);
                    else
                        queryString = queryString + "?location=" + encodeURIComponent(zip) + "&distance=" + encodeURIComponent(dist);
                }
                if (gen != "0" && gen != "B") {
                    if (queryString.indexOf("?") == -1) {
                        queryString = queryString + "?gender=" + encodeURIComponent(gen);
                    }
                    else {
                        queryString = queryString + "&gender=" + encodeURIComponent(gen);
                    }

                }
                 if (lang != 0 && lang != 2) { //lang 0 = "language spoken", 1- "English", 2- "All Languages"
                     if (queryString.indexOf("?") == -1) {
                         queryString = queryString + "?language=" + encodeURIComponent(lang);
                     }
                     else {
                        queryString = queryString + "&language=" + encodeURIComponent(lang);
                     }
                }
                if (ins != "Insurance Carrier") {
                    if (queryString.indexOf("?") == -1) {
                        queryString = queryString + "?insurance=" + encodeURIComponent(ins);
                    }
                    else {
                        queryString = queryString + "&insurance=" + encodeURIComponent(ins);
                    }
                }
                if ((gen == 'B') || (queryString.indexOf("?") != -1)) { //if gender = both allow search to continue or do not redirect without search criteria
                    window.location.assign(queryString);
                }

            }

             /*
             * The search by last name option is separate from the rest of the form; we don't need to post the whole thing.
             */
            function runNameSearch() {
                var name = $("#txtLName").val();
                if (name.length > 0 ) {
                    var queryString = searchResultsURL + "?name=" + encodeURIComponent(name);
                    window.location.assign(queryString);
                }
            }
            

