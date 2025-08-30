import types
import pytest

from bfm.core.registry import Registry

def test_register_and_resolve_class():
    reg = Registry()  # autodiscover unused here

    @reg.register("foo")
    class Foo:
        def __init__(self, x: int): self.x = x

    obj = reg.resolve("foo", x=3)
    assert isinstance(obj, Foo)
    assert obj.x == 3

def test_aliases_and_contains_and_list():
    reg = Registry() 
    
    @reg.register("bar", "b", "BA")
    class Bar:
        pass

    assert reg.contains("bar")
    assert reg.contains("b")
    # list() only returns canonical keys
    assert reg.list() == ["bar"]
    # alias mapping is lowercase → canonical
    assert reg.list_aliases() == {"b": "bar", "ba": "bar"}
    
def test_duplicate_key_raises():
    reg = Registry() 

    @reg.register("dup")
    class A: ...
    with pytest.raises(KeyError):
        @reg.register("dup")
        class B: ...
        
    with pytest.raises(KeyError):
        @reg.register("dup2", "dup2")
        class C: ...

def test_duplicate_alias_raises():
    reg = Registry()

    @reg.register("alpha", "a1")
    class A: ...
    with pytest.raises(KeyError):
        @reg.register("beta", "a1")
        class B: ...
        
    with pytest.raises(KeyError):
        @reg.register("gamma", "g1", "g1")
        class C: ...

def test_get_returns_factory_and_resolve_instantiates():
    reg = Registry()

    @reg.register("ctor")
    class C:
        def __init__(self, p=0): self.p = p

    fac = reg.get("ctor")
    assert fac is C
    obj = reg.resolve("ctor", p=7)
    assert isinstance(obj, C) and obj.p == 7