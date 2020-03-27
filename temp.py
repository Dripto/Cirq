import sys
print(sys.meta_path)

for finder in sys.meta_path:
    print(finder, hasattr(finder, 'find_spec'))

for finder in list(sys.meta_path):
    sys.meta_path = [finder]
    try:
        import cirq
    except ImportError:
        pass
    except BaseException as e:
        print('Other error', finder, e)
    else:
        print('Imported by', finder)
        break
