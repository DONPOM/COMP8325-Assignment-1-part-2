
                        $(document).ready(function() {
                            Pb.Form.addHash('#logoutForm');
                            $('.logoutLink').bind('click', function() {
                                $('#logoutForm').submit();
                            });
                            $('.upgradeLink').bind('click', function() {
                                Pb.Track.tr('navigation_upgrade_click');
                            });
                        });

                        (function() {
                            Pb.InitEventQueue.add(function() {
                                var jsonObj = {"username":"Shohae_Yu","isHashtagPartner":false};

                                Pb.Data.add(
                                    'loggedInUserData',
                                    jsonObj);
                            });
                        })();
                    