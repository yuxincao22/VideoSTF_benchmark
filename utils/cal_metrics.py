from metrics import RRMetric, RIMetric, IEMetric

rr = RRMetric(n=5, threshold=1)
ri = RIMetric(n=1)
ie = IEMetric(n=1)

def cal(text):
    rr_value = rr(text)
    ri_value = ri(text)
    ie_value = ie(text)
    return rr_value, ri_value, ie_value

def cal_batch(texts):
    rr_value = rr(texts).value
    ri_value = ri(texts).value
    ie_value = ie(texts).value
    return rr_value, ri_value, ie_value