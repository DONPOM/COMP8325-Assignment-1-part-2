//hahaha
(function () {
    var LoginDialog = window.LoginDialog || {};
    LoginDialog.Resources = {
      fontFaces: ['HelveticaNeueW01-45Ligh', 'HelveticaNeueW01-55Roma', 'HelveticaNeueW01-65Medi', 'HelveticaNeueW01-75Bold'],
      images: ['resources/images/loading.gif','resources/images/black-x.png', 'resources/images/blue-btn-sprite.png', 'resources/images/forgot-password.png', 'resources/images/btn_red_32.png', 'resources/images/fb-f2.png']
    };
    LoginDialog.View = {
        create : function (messages, hideCloseButton) {
            var html =
                '<div class="dialog-body">\
                                <div id="update-to-sp3-message">\
                                  <div id="update-to-sp3-message-icon"></div>\
                                  <span id="update-to-sp3-message-text">\
                                    '+ messages.update_to_sp3_message.replace('$url',messages.update_to_sp3_link) +'\
                                  </span>\
                                </div>\
                    '+(hideCloseButton?'<a href="#" tabindex="10" class="dialog-close action-close"></a>':'')+'\
                    <div class=\'loading loading-icon\' />\
                    <div class=\'loading loading-mask\' />\
                    <div id="left">\
                      <form class="dialog-tab dialog-box-login" target="#" style="display:none">\
                          <div class="clearfix"></div>\
                          <div class="dialog-box">\
                              <div class="dialog-title">' + messages.login_title + '</div>\
                          </div>\
                          <div class="rightsep">\
                            <div class="input-box text-input-box">\
                                <input autofocus tabindex="1" id="login-input-email" name="email" type="text" maxlength="60" class="inp enter-key" validation="notempty;email" placeholder="' + messages.label_email + '" />\
                                <label name="email" class="input-box-desc hidden-element error-message"></label>\
                                <label class="input-box-desc desc-message">' + messages.description_email + '</label>\
                            </div>\
                            <div class="input-box radio-input-box">\
                                <input type="radio" tabindex="7" id="context-switch-radio-sign-up" name="context" value="'+ LoginDialog.Contexts.SIGN_UP + '">\
                                <label for="context-switch-radio-sign-up">' + messages.label_new_user + '</label><br/>\
                                <input type="radio" tabindex="7" id="context-switch-radio-login" name="context" value="' + LoginDialog.Contexts.LOGIN + '" checked>\
                                <label for="context-switch-radio-login">' + messages.label_existing_user + '</label>\
                            </div>\
                            <div class="input-box text-input-box" style="margin-bottom: 8px">\
                                <input tabindex="2" id="login-input-password" name="password" type="password" maxlength="15" class="inp enter-key" validation="notempty;length:{4,15};password" placeholder="' + messages.label_password + '" />\
                                <label name="password" class="input-box-desc hidden-element error-message"></label>\
                            </div>\
                            <div class="dialog-box">\
                                <span class="dialog-checkbox">\
                                   <input tabindex="5" id="login-input-remember" name="rememberMe" type="checkbox" checked="checked" />\
                                   <label for="login-input-remember">' + messages.label_remember_me + '</label>\
                                   <a id="login-switch-context-forgot-password" href="#" class="switch-context switch-context-link right" rev="' + LoginDialog.Contexts.FORGOT_PASSWORD + '">' + messages.forgot_password + '</a>\
                                </span>\
                            </div>\
                            <div class="clearfix"></div>\
                            <div id="captcha-wrapper-login" class="hidden">\
                                <span class="captcha-description">'+ messages.captcha_description +'</span>\
                                <div id="wix-captcha-default-login"></div>\
                                <label name="captchaValidation"></label>\
                            </div>\
                            <div class="action-box">\
                                <input type="submit" href="#" tabindex="6" id="login-action-go" class="dialog-button action-login" value="' + messages.label_button_go + '">\
                                <a href="#" tabindex="6" tabindex="10" class="skip-login action-close switch-context-link" style="display:none">' + messages.skip_login + '</a>\
                            </div>\
                          </div>\
                      </form>\
                      <form class="dialog-tab dialog-box-sign-up" style="display:none">\
                          <div class="clearfix"></div>\
                          <div class="dialog-box">\
                              <div class="dialog-title">' + messages.sign_up_title + '</div>\
                          </div>\
                          <div class="rightsep">\
                            <div class="input-box text-input-box">\
                                <span class="input-box-title sign-up-email-title">' + messages.sign_up_title_email + '</span>\
                                <input tabindex="1" id="sign-up-input-email" name="email" type="text" maxlength="60" class="inp enter-key" validation="notempty;email" placeholder="' + messages.label_email + '" />\
                                <label name="email" class="input-box-desc hidden-element error-message"></label>\
                                <div class="padding-box"></div>\
                                <input tabindex="2" id="sign-up-input-email-validation" name="emailValidation" type="text" maxlength="60" class="inp enter-key" validation="match:email:trim,lowerCase" autocomplete="off" placeholder="' + messages.label_validate_email + '" />\
                                <label name="emailValidation" class="input-box-desc hidden-element error-message"></label>\
                                <label class="input-box-desc desc-message">' + messages.description_email + '</label>\
                          </div>\
                          <div class="input-box text-input-box">\
                              <span class="input-box-title sign-up-password-title">' + messages.sign_up_title_password + '</span>\
                              <input tabindex="3" id="sign-up-input-password" name="password" type="password" maxlength="15" class="inp enter-key" validation="notempty;length:{4,15};password" autocomplete="off" placeholder="' + messages.label_password + '" />\
                              <label name="password" class="input-box-desc hidden-element error-message"></label>\
                              <div class="padding-box"></div>\
                              <input tabindex="4" id="sign-up-input-password-validation" name="passwordValidation" type="password" maxlength="15" validation="match:password" class="inp enter-key" autocomplete="off" placeholder="' + messages.label_validate_password + '" />\
                              <label name="passwordValidation" class="input-box-desc hidden-element error-message"></label>\
                          </div>\
                          <div id="captcha-wrapper-sign-up" class="hidden">\
                                <span class="captcha-description">'+ messages.captcha_description +'</span>\
                                <div id="wix-captcha-default-sign-up"></div>\
                                <label name="captchaValidation"></label>\
                            </div>\
                            <div class="dialog-box">\
                                <a id="sign-up-switch-context-login" href="#" class="switch-context switch-context-link" rev="' + LoginDialog.Contexts.LOGIN + '">&lt;&lt;&nbsp;' + messages.label_existing_user + '</a>\
                            </div>\
                            <div class="action-box">\
                                <input type="submit" id="sign-up-action-go" href="#" tabindex="5" class="dialog-button action-signup" value="' + messages.sign_up_button + '">\
                            </div>\
                          </div>\
                          <div class="dialog-box">\
                              <label class="input-box-desc desc-message">\
                                  ' + messages.sign_up_terms_start + '\
                                  <a href="' + messages.sign_up_user_terms_url + '" target="_blank">' + messages.sign_up_user_terms_link + '</a>,\
                                  <a href="' + messages.sign_up_privacy_terms_url + '" target="_blank">' + messages.sign_up_privacy_terms_link + '</a>\
                                  ' + messages.sign_up_terms_end + '\
                              </label>\
                          </div>\
                      </form>\
                      <form class="dialog-tab dialog-box-set-password" style="display:none">\
                          <div class="clearfix"></div>\
                          <div class="dialog-box">\
                              <div class="dialog-title">' + messages.set_password_title + '</div>\
                          </div>\
                          <div class="forgot-desc">' + messages.set_password_description + '</div>\
                          <div class="input-box text-input-box">\
                            <input tabindex="1" id="set-password-input-password" name="password" type="password" maxlength="15" class="inp enter-key" validation="notempty;length:{4,15};password" placeholder="' + messages.label_set_password + '" />\
                            <label name="password" class="input-box-desc hidden-element error-message"></label>\
                            <div class="padding-box"></div>\
                            <input tabindex="2" id="set-password-input-password-validation" name="passwordValidation" type="password" maxlength="15" validation="match:password" class="inp enter-key" placeholder="' + messages.label_validate_password + '" />\
                            <label name="passwordValidation" class="input-box-desc hidden-element error-message"></label>\
                          </div>\
                          <div class="action-box">\
                              <input type="submit" id="set-password-action-go" href="#" class="dialog-button action-set" value="' + messages.label_button_go + '">\
                          </div>\
                      </form>\
                      <form class="dialog-tab dialog-box-forgot-password" style="display:none">\
                          <div class="step-1">\
                              <div class="clearfix"></div>\
                              <div class="dialog-box">\
                                  <div class="dialog-title">' + messages.forgot_password_title + '</div>\
                              </div>\
                              <div class="forgot-desc">' + messages.forgot_password_description + '</div>\
                              <div class="input-box text-input-box">\
                                  <input tabindex="1" id="forgot-password-input-email" name="email" type="text" maxlength="60" class="inp enter-key" validation="notempty;email" placeholder="' + messages.label_email + '" />\
                                  <label name="email" class="input-box-desc hidden-element error-message"></label>\
                                  <label class="input-box-desc desc-message">' + messages.description_email + '</label>\
                              </div>\
                              <div class="dialog-box">\
                                  <a href="#" id="forgot-password-switch-context-login" class="switch-context switch-context-link switch-context-label" rev="' + LoginDialog.Contexts.LOGIN + '">&lt;&lt;&nbsp;' + messages.back_to_login + '</a>\
                              </div>\
                              <div class="action-box">\
                                  <input type="submit" id="forgot-password-action-go" href="#" class="dialog-button action-forgot" value="' + messages.label_button_go + '">\
                              </div>\
                          </div>\
                          <div class="step-2 hidden-element">\
                              <div class="clearfix"></div>\
                              <div class="dialog-box forgot-done-box">\
                                  <span class="forgot-done-img"></span>\
                                  <span class="forgot-done-desc">' + messages.forgot_password_done + '</span>\
                              </div>\
                              <div class="clearfix"></div>\
                              <div class="action-box">\
                                  <a id="forgot-password-action-done" href="#" class="dialog-button action-close">' + messages.done + '</a>\
                              </div>\
                          </div>\
                      </form>\
                      <form class="dialog-tab dialog-box-account-verify" style="display:none">\
                          <div class="step-1">\
                              <div class="clearfix"></div>\
                              <div class="dialog-box">\
                                  <div class="dialog-title">' + messages.account_verify_title + '</div>\
                              </div>\
                              <div class="description"><span class="email bold description"></span> ' + messages.account_verify_description + '</div>\
                              <div class="description">' + messages.account_verify_description_b + '</div>\
                              <div class="input-box text-input-box">\
                                <input tabindex="2" id="account-verify-input-password" name="password" type="password" maxlength="15" class="inp enter-key" validation="notempty;length:{4,15};password" placeholder="' + messages.label_password_verify + '" />\
                                <label name="password" class="input-box-desc hidden-element error-message"></label>\
                                <span class="input-box-desc desc-message"></span>\
                              </div>\
                              <div class="dialog-box">\
                                  <a id="verify-switch-context-forgot-password" href="#" class="switch-context switch-context-link" rev="' + LoginDialog.Contexts.FORGOT_PASSWORD + '">' + messages.forgot_password + '</a>\
                                  <div class="clearfix"></div>\
                              </div>\
                              <div class="action-box">\
                                  <input type="submit" id="forgot-password-action-go" tabindex="5" href="#" class="dialog-button action-verify" value="' + messages.label_button_go + '">\
                              </div>\
                          </div>\
                      </form>\
                      <form class="dialog-tab dialog-box-account-merge" style="display:none">\
                          <div class="clearfix"></div>\
                          <div class="dialog-box">\
                              <div class="dialog-title">' + messages.merge_title + '</div>\
                          </div>\
                          <div class="description" social_network_str="' + messages.merge_description + '"></div>\
                          <div class="bold description" social_network_str="' + messages.merge_description_bold + '"></div>\
                          <div class="input-box text-input-box">\
                              <input autofocus tabindex="1" id="account-merge-input-email" name="email" type="text" maxlength="60" class="inp enter-key" validation="notempty;email" placeholder="' + messages.label_email + '" />\
                              <label name="email" class="input-box-desc hidden-element error-message"></label>\
                          </div>\
                          <div class="input-box text-input-box">\
                              <input tabindex="2" id="account-merge-input-password" name="password" type="password" maxlength="15" class="inp enter-key" validation="notempty;length:{4,15};password" placeholder="' + messages.label_password + '" />\
                              <label name="password" class="input-box-desc hidden-element error-message"></label>\
                          </div>\
                          <div class="dialog-box">\
                              <a id="merge-switch-context-forgot-password" href="#" class="switch-context switch-context-link" rev="' + LoginDialog.Contexts.FORGOT_PASSWORD + '">' + messages.forgot_password + '</a>\
                          </div>\
                          <div class="clearfix"></div>\
                          <div class="action-box">\
                              <input type="button" href="#" tabindex="7" id="merge-action-no" class="dialog-button action-nomerge" value="' + messages.label_button_no + '">\
                              <input type="submit" href="#" tabindex="6" id="merge-action-go" class="dialog-button action-merge" value="' + messages.label_button_ok + '">\
                          </div>\
                      </form>\
                      <form class="dialog-tab dialog-box-get-email" style="display:none">\
                          <div class="clearfix"></div>\
                          <div class="dialog-box">\
                              <div class="dialog-title small">' + messages.get_email_title + '</div>\
                          </div>\
                          <div class="input-box text-input-box">\
                              <input autofocus tabindex="1" id="account-merge-input-email" name="email" type="text" maxlength="60" class="inp enter-key" validation="notempty;email" placeholder="' + messages.label_email + '" />\
                              <label name="email" class="input-box-desc hidden-element error-message"></label>\
                              <label class="input-box-desc desc-message">' + messages.description_email + '</label>\
                          </div>\
                          <div class="clearfix"></div>\
                          <div class="action-box">\
                              <input type="submit" href="#" tabindex="2" id="get-email-action-go" class="dialog-button action-get-email" value="' + messages.label_button_ok + '">\
                          </div>\
                      </form>\
                    </div>\
                    <div id="right">\
                      <div class="right-slider">\
                        <div class="social-login-tab-container">\
                        <div class="seperator-text">' + messages.social_sign_in_seperator + '</div>\
                          <div class="social-login-tab clearfix">\
                            <div class="social-desc">' + messages.google_sign_in_desc + '</div>\
                            <table>\
                              <tr class="social-button-container">\
                                <td>\
                                    <div id="customFBBtn" class="buttonText"><span class="icon"><span></span></span>' + messages.facebook_sign_in_button + '</div>\
                                    <label name="fb" class="input-box-desc hidden-element error-message"></label>\
                                </td>\
                              </tr>\
                              <tr class="social-button-container">\
                                <td>\
                                    <div id="customGPBtn" class="buttonText"><span class="icon"><span></span></span>' + messages.google_sign_in_button + '</div>\
                                    <label name="gapi" class="input-box-desc hidden-element error-message"></label>\
                                </td>\
                              </tr>\
                            </table>\
                          </div>\
                        </div>\
                      </div>\
                    </div>\
                </div>';

            return html;
        }
    }
})();