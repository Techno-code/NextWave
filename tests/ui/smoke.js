// UI smoke test: drives static/index.html in jsdom against a live server (see run.sh). Needs node and jsdom.
const {JSDOM}=require("jsdom"),fs=require("fs"),B="http://127.0.0.1:8765";
const html=fs.readFileSync(process.env.REPO+"/static/index.html","utf8");
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const dom=new JSDOM(html,{runScripts:"dangerously",url:B+"/",pretendToBeVisual:true,beforeParse(w){w.fetch=(u,o)=>fetch(new URL(u,B),o);w.URL.createObjectURL=()=>"blob:x";w.scrollTo=()=>{};w.alert=m=>console.log("ALERT",m);w.confirm=()=>true}});
const w=dom.window,d=w.document,$=s=>d.querySelector(s);let fails=0;const SS=()=>w.eval("S"),PP=()=>w.eval("P");
const T0=Date.now();const ok=(c,m)=>{console.log(((Date.now()-T0)/1000).toFixed(1)+"s "+(c?"PASS ":"FAIL ")+m.slice(0,110));if(!c)fails++};
const waitFor=async(f,ms=15000)=>{const t=Date.now();while(Date.now()-t<ms){try{const v=f();if(v)return v}catch(e){}await sleep(150)}return null};
(async()=>{
 await waitFor(()=>!$("#gate").hidden);
 $("#acc").checked=true;$("#acc").dispatchEvent(new w.Event("change"));$("#go").click();
 await waitFor(()=>$("#su")); ok(!!$("#ct"),"register form shows the consent checkbox");
 $("#em").value="ui@example.com";$("#pw").value="password1";
 $("#su").click(); await sleep(800); ok(/18 or older/.test($("#view").textContent)&&!SS().token,"create account without ticking the box is refused");
 $("#ct").checked=true;$("#su").click(); await waitFor(()=>SS().token); ok(!!SS().token&&!SS().needConsent,"account created with explicit acceptance");
 await waitFor(()=>/What do you need to decide/.test($("#view").textContent)); ok(/What do you need to decide/.test($("#view").textContent)&&d.querySelectorAll(".door").length===4,"signed-in users land on Start here with four plain-language choices");
 ok([...d.querySelectorAll("#tabs button")].map(b=>b.textContent)[0]==="Start here"&&![...d.querySelectorAll("#tabs button")].some(b=>b.textContent==="Problem helper"),"Start here is the first tab and the offline helper is hidden when signed in");
 ok(/How to read an answer/.test($("#view").textContent)&&/Plain-English glossary/.test($("#view").textContent),"Start page explains confidence, checks, tiers and key words");
 d.querySelector('[data-g="sol"]').click(); await waitFor(()=>$("#svg")); ok(SS().tab==="sol"&&!!$("#svg"),"a Start choice opens the matching page");
 SS().tab="start";w.nav();w.render();
 // terms version bump simulated by revoking, then re-consent through the UI
 SS().tab="trust";w.nav();w.render(); await waitFor(()=>$("#view").textContent.includes("Your account and rights"));
 ok($("#view").textContent.includes("Your account and rights")&&!$("#view").textContent.includes("Show my data"),"Trust tab shows server-backed account rights and hides the local-only duplicates");
 const x=d.querySelector('[data-a="x"]'); ok(!!x,"export button present");
 await fetch(B+"/api/v1/consent/revoke",{method:"POST",headers:{Authorization:"Bearer "+SS().token}});
 SS().tab="ask";w.nav();w.render();
 const r=await w.fetch("/api/v1/feeds",{headers:{Authorization:"Bearer "+SS().token}}); await waitFor(()=>SS().needConsent);
 ok(r.status===403&&!!SS().needConsent,"a blocked request switches the UI to the re-consent view");
 await waitFor(()=>$("#c3")); ok(!!$("#c3")&&$("#view").textContent.includes("Download my data"),"re-consent view offers acceptance and still shows account rights");
 $("#c1").checked=true;$("#c1").dispatchEvent(new w.Event("change"));$("#c3").click(); await waitFor(()=>!SS().needConsent); ok(!SS().needConsent,"accepting restores access");
 // planner cards
 SS().tab="opt";w.nav();w.render(); await waitFor(()=>d.querySelector('[data-p="go"]')); ok(!!d.querySelector('[data-f="go"]')&&!!d.querySelector('[data-p="go"]'),"planner shows feeds and tiers/channels cards");
 await sleep(500);
 d.querySelector('[data-p="t"]').checked=true;d.querySelector('[data-p="ch"]').checked=true;d.querySelector('[data-p="b"]').value="anneal";d.querySelector('[data-p="go"]').click();
 await waitFor(()=>/Backends against the exact reference/.test($("#view").textContent)); const t=d.querySelector('[data-p="out"]').textContent;
 ok(/Exact reference proven optimal/.test(t)&&/simulated annealing/i.test(t)&&/Verified: all constraints pass/.test(t),"tier+channel plan runs on the annealing backend and is verified: "+t.slice(0,90));
 ok(/Test|Standard|Full/.test(t)&&/Organic|Paid social|Creator/.test(t),"plan table lists tier and channel per funded trend");
 d.querySelector('[data-p="b"]').value="hardware";d.querySelector('[data-p="b"]').dispatchEvent(new w.Event("change")); ok(d.querySelector('[data-p="b"]').querySelector('[value="hardware"]').disabled,"hardware backend is disabled when the server has none configured");
 // feed via URL through the UI
 d.querySelector('[data-f="un"]').value="local csv";d.querySelector('[data-f="uu"]').value="http://127.0.0.1:8801/feed.csv";d.querySelector('[data-f="ub"]').click();
 await waitFor(()=>/Feed added/.test(d.querySelector('[data-f="msg"]').textContent)); ok(/local csv/.test(d.querySelector('[data-f="list"]').textContent)&&/days old/.test(d.querySelector('[data-f="list"]').textContent),"URL feed added and shown with a staleness badge");
 d.querySelector('[data-f="th"]').value="2";d.querySelector('[data-f="go"]').click();
 await waitFor(()=>d.querySelector('[data-f="out"] table')); ok(d.querySelectorAll('[data-f="out"] tbody tr').length===10&&/days old/.test(d.querySelector('[data-f="out"]').textContent),"forecast from the feed returns 10 trends and the stale-data warning");
 d.querySelector('[data-f="use"]').click(); await waitFor(()=>SS().tab==="opt"&&/trend \d+/.test($("#view").textContent)); ok(PP().T.length===10&&/^trend /.test(PP().T[0].n),"'Use these in the planner' loads the feed's trends");
 // find and solve tab
 SS().tab="sol";w.nav();w.render(); await waitFor(()=>$("#svg")); let so; ok(!!$("#svg")&&!!$("#svx")&&!!$("#scb"),"Find and solve tab shows the problem box, example loader and catalogue search");
 $("#svq").value="";$("#svi").value="";$("#svg").click(); await sleep(200); ok(/Describe the problem in a sentence first/.test($("#svm").textContent),"pressing the button with nothing entered gives guidance instead of an error");
 d.querySelector('[data-s="1"]').click(); await waitFor(()=>/Hungarian/.test($("#svo").textContent),20000); so=$("#svo").textContent;
 ok(/Ana|Ben|Chen/.test(so)&&/Checked: \d+ of \d+ checks passed/.test(so)&&d.querySelectorAll("#svo .big").length===1,"a ready-made example runs in one click and shows a readable table and a plain checked line");
 ok(d.querySelectorAll("#svo details").length>=3&&d.querySelector("#svo h2")===d.activeElement,"technical detail is collapsed and focus moves to the answer");
 $("#svf").value="";$("#svi").value="";$("#svc").value="";$("#svq").value="Choose which projects to fund under a capital limit";$("#svd").checked=true;$("#svg").click();
 await waitFor(()=>/Playbook PB-/.test($("#svo").textContent),20000); so=$("#svo").textContent;
 ok(/Verified: yes/.test(so)&&/illustrative/i.test(so)&&/Theorems behind it/.test(so)&&/References/.test(so),"illustrative solve is verified, labelled illustrative, and cites theorems and references");
 ok(/The detail behind it/.test(so)&&/How firm it is/.test(so)&&/What if one thing changed/.test(so),"the detailed answer, its firmness and the what-if table are shown");
 ok(d.querySelectorAll("#svo table").length>=1&&[...d.querySelectorAll("#svo th")].some(h=>h.textContent==="Check"),"checks are shown in a table with headers");
 $("#svd").checked=false;$("#svf").value="assignment";$("#svx").click(); await waitFor(()=>/cost/.test($("#svi").value)); ok(/"cost"/.test($("#svi").value),"example loader fills the numbers for the chosen type");
 $("#svg").click(); await waitFor(()=>/Hungarian/.test($("#svo").textContent),20000); so=$("#svo").textContent; ok(/Hungarian/.test(so)&&/Verified: yes/.test(so)&&/Independent solve by mixed-integer programming/.test(so),"assignment example solved and cross-checked by a second method");
 $("#svi").value="{bad json";$("#svg").click(); await sleep(300); ok(/not valid JSON/.test($("#svm").textContent),"invalid JSON is explained, not sent");
 $("#svf").value="";$("#svi").value=JSON.stringify({values:[1,2],weights:[1],capacity:3});$("#svg").click(); await waitFor(()=>/expected 2/.test($("#svo").textContent)); ok(/expected 2/.test($("#svo").textContent),"a bad instance returns a plain-language error in the page");
 $("#svi").value="";$("#svf").value="";$("#svc").value="";$("#svq").value="Budget of 10\n- A: cost 4 value 7\n- B: cost 5 value 9\n- C: cost 6 value 10";$("#svg").click(); await waitFor(()=>/Quantum simulator cross-check/.test($("#svo").textContent),20000); so=$("#svo").textContent;
 ok(/Choose A, C/.test(so)&&/match/.test(so)&&/No quantum advantage/.test(so),"options typed in the text are read, solved exactly, and cross-checked on the QAOA simulator");
 $("#scq").value="assign nurses to shifts";$("#scb").click(); await waitFor(()=>/No similar problems found|<table/.test($("#sco").innerHTML)||/index has not been built/.test($("#sco").textContent)); ok(d.querySelectorAll("#sco table tbody tr").length>0||/index has not been built/.test($("#sco").textContent),"catalogue search lists problems, or says the index is not built");
 // appeal + export + delete
 SS().tab="trust";w.nav();w.render(); await waitFor(()=>d.querySelector('[data-a="ab"]'));
 d.querySelector('[data-a="ak"]').value="account";d.querySelector('[data-a="ak"]').dispatchEvent(new w.Event("change"));d.querySelector('[data-a="aw"]').value="My account was restricted in error.";d.querySelector('[data-a="ab"]').click();
 await waitFor(()=>/Appeal filed/.test(d.querySelector('[data-a="am"]').textContent)); await waitFor(()=>/restricted in error/.test(d.querySelector('[data-a="al"]').textContent)); ok(/restricted in error/.test(d.querySelector('[data-a="al"]').textContent),"appeal filed and listed");
 SS().tab="ask";w.nav();w.render(); await waitFor(()=>$("#q")); $("#q").value="how do I launder money";$("#run").click(); await waitFor(()=>d.querySelector('[data-rf="b"]')); ok(!!d.querySelector('[data-rf="b"]'),"a refused request can be appealed from the Ask page");
 d.querySelector('[data-rf="w"]').value="I work in bank compliance and need to understand detection.";d.querySelector('[data-rf="b"]').click(); await waitFor(()=>/Review requested/.test(d.querySelector('[data-rf="m"]').textContent)); ok(/Review requested/.test(d.querySelector('[data-rf="m"]').textContent),"refusal appeal is accepted");
 SS().tab="trust";w.nav();w.render(); await waitFor(()=>d.querySelector('[data-a="al"]')&&/refusal/.test(d.querySelector('[data-a="al"]').textContent)); ok(/refusal/.test(d.querySelector('[data-a="al"]').textContent),"refusal appeal is listed under Privacy and data");
 d.querySelector('[data-a="dp"]').value="password1";d.querySelector('[data-a="dc"]').value="nope";d.querySelector('[data-a="d"]').click(); await waitFor(()=>/exactly/.test(d.querySelector('[data-a="dm"]').textContent)); ok(/exactly/.test(d.querySelector('[data-a="dm"]').textContent)&&!!SS().token,"deletion needs the exact phrase");
 d.querySelector('[data-a="dc"]').value="DELETE MY ACCOUNT";d.querySelector('[data-a="d"]').click(); await waitFor(()=>/Account deleted/.test($("#view").textContent)); ok(/Account deleted/.test($("#view").textContent)&&!SS().token,"account deleted from the UI");
 console.log(fails?fails+" FAILED":"ALL UI CHECKS PASSED"); process.exit(fails?1:0)})().catch(e=>{console.log("ERR",e);process.exit(2)});
