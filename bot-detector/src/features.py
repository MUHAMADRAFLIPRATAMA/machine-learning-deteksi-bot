"""Fitur akun, konten, dan koordinasi."""
import re
import numpy as np
import pandas as pd
URL=re.compile(r"https?://\S+",re.I); MENTION=re.compile(r"@\w+"); HASHTAG=re.compile(r"#\w+"); RT=re.compile(r"^\s*RT\s+@?(\w+)",re.I)
def content_features(text):
 s=text.fillna("").astype(str)
 return pd.DataFrame({"text_len":s.str.len(),"word_count":s.str.split().str.len(),"url_count":s.map(lambda x:len(URL.findall(x))),"mention_count":s.map(lambda x:len(MENTION.findall(x))),"hashtag_count":s.map(lambda x:len(HASHTAG.findall(x))),"is_retweet":s.str.match(RT).astype(int)},index=text.index)
def account_features(df,text=None):
 out=pd.DataFrame(index=df.index)
 def col(names): return next((c for c in names if c in df),None)
 def num(names):
  c=col(names); return pd.to_numeric(df[c],errors="coerce").fillna(0) if c else None
 followers=num(["followers_count","followers","follower_count"]); following=num(["friends_count","following","following_count"]); statuses=num(["statuses_count","statuses","tweet_count","posts"])
 for n,v in (("followers",followers),("following",following),("statuses",statuses)):
  if v is not None: out[n]=v
 if followers is not None and following is not None: out["follow_ratio"]=followers/(following+1)
 created=col(["created_at","account_creation","creation_date"])
 if created:
  dt=pd.to_datetime(df[created],errors="coerce",utc=True); ref=dt.max() if dt.notna().any() else pd.Timestamp.now(tz="UTC"); age=(ref-dt).dt.total_seconds().div(86400).fillna(1).clip(lower=1); out["age_days"]=age
  if statuses is not None: out["posts_per_day"]=statuses/age
 verified=col(["verified","is_verified"])
 if verified: out["verified"]=df[verified].astype(str).str.lower().isin(["true","1","yes"]).astype(int)
 bio=col(["description","bio","profile_description"])
 if bio:
  b=df[bio].fillna("").astype(str); out["bio_empty"]=(b.str.strip()=="").astype(int); out["bio_len"]=b.str.len()
 if text is not None: out=pd.concat([out,content_features(text)],axis=1)
 return out.replace([np.inf,-np.inf],0).fillna(0)
def coordination_features(df,text_col="text",time_col=None,window_minutes=10):
 if text_col not in df:return pd.DataFrame(index=df.index)
 x=df[text_col].fillna("").astype(str).str.lower().str.replace(r"https?://\S+","URL",regex=True).str.replace(r"\s+"," ",regex=True).str.strip(); out=pd.DataFrame(index=df.index)
 out["copy_paste_count"]=x.map(x.value_counts()).sub(1).clip(lower=0)
 tags=x.map(lambda s:" ".join(sorted(set(HASHTAG.findall(s))))); out["co_hashtag_count"]=tags.map(tags.value_counts()).sub(1).where(tags!="",0).fillna(0).clip(lower=0)
 targets=x.map(lambda s:(RT.search(s).group(1) if RT.search(s) else "")); out["co_retweet_count"]=targets.map(targets.value_counts()).sub(1).where(targets!="",0).fillna(0).clip(lower=0)
 if time_col and time_col in df:
  t=pd.to_datetime(df[time_col],errors="coerce",utc=True).dt.floor(f"{window_minutes}min"); out["same_window_count"]=t.map(t.value_counts()).sub(1).fillna(0).clip(lower=0)
 else: out["same_window_count"]=0
 return out
